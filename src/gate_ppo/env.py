"""Functional gate-racing environment on crazyflow's ``SimData`` (RPM actions, batched worlds).

The environment is a pair of pure functions ``reset`` / ``step`` over an :class:`EnvState` pytree
that wraps crazyflow's ``SimData``. Physics runs through ``sim.build_step_fn()`` (10 dynamics ticks
per 48 Hz control step), gate events are geometric tests in JAX (no MJX contact queries), and worlds
auto-reset in the same step so the whole rollout can live inside ``jax.lax.scan``.

Observation (37-D, all in the drone's body frame so the policy is yaw-invariant):
``z (1) | gravity dir (3) | vel (3) | ang_vel (3) | rotor_vel normalised (4) | prev action (4) |
current gate: rel pos (3), normal (3), lateral axis (3) | next gate: same (9) |
gates remaining (1)``.

Action: ``a in [-1, 1]^4`` -> ``rpm = motor_gain * hover_rpm * (1 + action_scale * a)``. The
commanded RPM (``controls.rotor_vel``) is the label the VLA will later be trained on;
``states.rotor_vel`` is the realised rotor speed after crazyflow's first-order rotor dynamics.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.dynamics.first_principles import dynamics as fp_dynamics
from crazyflow.sim import Sim
from crazyflow.sim import functional as F
from crazyflow.sim.data import SimData
from crazyflow.utils import leaf_replace
from flax import struct
from jax import Array

from gate_ppo.config import EnvConfig
from gate_ppo.geometry import gate_crossing, gate_frame, hover_rpm, quat_rotate_inv
from gate_ppo.track import sample_start, sample_track

OBS_DIM = 37
ACTION_DIM = 4


@struct.dataclass
class EnvParams:
    """Traced environment parameters: change them between iterations without recompiling."""

    yaw_range: Array
    n_gates_min: Array
    n_gates_max: Array
    dr_mass: Array
    dr_inertia: Array
    dr_motor_gain: Array

    @classmethod
    def from_config(
        cls,
        cfg: EnvConfig,
        yaw_range: float | None = None,
        n_gates_min: int | None = None,
        n_gates_max: int | None = None,
        domain_randomization: bool = True,
    ) -> EnvParams:
        dr = 1.0 if domain_randomization else 0.0
        return cls(
            yaw_range=jnp.float32(cfg.yaw_range if yaw_range is None else yaw_range),
            n_gates_min=jnp.int32(cfg.n_gates_min if n_gates_min is None else n_gates_min),
            n_gates_max=jnp.int32(cfg.n_gates if n_gates_max is None else n_gates_max),
            dr_mass=jnp.float32(cfg.dr_mass * dr),
            dr_inertia=jnp.float32(cfg.dr_inertia * dr),
            dr_motor_gain=jnp.float32(cfg.dr_motor_gain * dr),
        )


@struct.dataclass
class EnvState:
    sim: SimData
    gates_pos: Array  # (N, G, 3), G = n_gates + 1 (the last one is a continuation gate)
    gates_yaw: Array  # (N, G)
    n_active: Array  # (N,) gates required for success
    gate_idx: Array  # (N,) index of the current target gate
    prev_pos: Array  # (N, 3)
    prev_action: Array  # (N, 4)
    t: Array  # (N,) control steps since reset
    ep_return: Array  # (N,)
    motor_gain: Array  # (N,)
    rng: Array


class GateRaceEnv:
    """Batched gate-racing task: holds the crazyflow ``Sim`` and constants; state is a pytree."""

    def __init__(self, cfg: EnvConfig, device: str = "cpu"):
        if cfg.sim_freq % cfg.ctrl_freq != 0:
            raise ValueError("sim_freq must be a multiple of ctrl_freq")
        if not 1 <= cfg.n_gates_min <= cfg.n_gates:
            raise ValueError("need 1 <= n_gates_min <= n_gates")
        self.cfg = cfg
        self.n_envs = cfg.n_envs
        self.n_gates = cfg.n_gates
        self.n_total_gates = cfg.n_gates + 1
        self.n_ticks = cfg.sim_freq // cfg.ctrl_freq
        self.sim = Sim(
            n_worlds=cfg.n_envs,
            n_drones=1,
            drone=cfg.drone,
            dynamics=Dynamics.first_principles,
            control=Control.rotor_vel,
            freq=cfg.sim_freq,
            device=device,
        )
        self.sim.reset()
        self._step_fn = self.sim.build_step_fn()
        params = load_params(fp_dynamics, cfg.drone)
        self.mass0 = float(np.asarray(params["mass"]))
        self.J0 = np.asarray(params["J"], dtype=np.float32)
        self.hover_rpm = float(hover_rpm(np.asarray(params["rpm2thrust"]), self.mass0))
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM

    # ----------------------------------------------------------------------------------- state
    def init_state(self, key: Array) -> EnvState:
        """Allocate an ``EnvState`` from the freshly reset ``Sim``; call :meth:`reset` next."""
        n, g = self.n_envs, self.n_total_gates
        return EnvState(
            sim=self.sim.data,
            gates_pos=jnp.zeros((n, g, 3)),
            gates_yaw=jnp.zeros((n, g)),
            n_active=jnp.full((n,), self.n_gates, dtype=jnp.int32),
            gate_idx=jnp.zeros((n,), dtype=jnp.int32),
            prev_pos=jnp.zeros((n, 3)),
            prev_action=jnp.zeros((n, ACTION_DIM)),
            t=jnp.zeros((n,), dtype=jnp.int32),
            ep_return=jnp.zeros((n,)),
            motor_gain=jnp.ones((n,)),
            rng=key,
        )

    def reset(
        self, state: EnvState, params: EnvParams, mask: Array | None = None
    ) -> tuple[EnvState, Array]:
        """Re-initialise the worlds selected by ``mask`` (all if None). Returns ``(state, obs)``."""
        mask = jnp.ones((self.n_envs,), dtype=bool) if mask is None else mask
        rng, key = jax.random.split(state.rng)
        state = self._reset_worlds(state.replace(rng=rng), mask, key, params)
        return state, self.observe(state)

    def _reset_worlds(
        self, state: EnvState, mask: Array, key: Array, params: EnvParams
    ) -> EnvState:
        cfg, n, g = self.cfg, self.n_envs, self.n_total_gates
        k_track, k_active, k_start, k_dr = jax.random.split(key, 4)

        track_fn = partial(
            sample_track,
            n_gates=g,
            yaw_range=params.yaw_range,
            spacing_min=cfg.spacing_min,
            spacing_max=cfg.spacing_max,
            dz_max=cfg.dz_max,
            z_min=cfg.z_min,
            z_max=cfg.z_max,
            start_z_min=cfg.start_z_min,
            start_z_max=cfg.start_z_max,
        )
        gates_pos, gates_yaw, start_pos, start_yaw = jax.vmap(track_fn)(
            jax.random.split(k_track, n)
        )
        n_active = jax.random.randint(
            k_active, (n,), params.n_gates_min, params.n_gates_max + 1, dtype=jnp.int32
        )
        pos, quat, vel, ang_vel, rotor_vel = sample_start(
            k_start,
            start_pos,
            start_yaw,
            hover_rpm=self.hover_rpm,
            xy_jitter=cfg.start_xy_jitter,
            yaw_noise=cfg.start_yaw_noise,
            vel_max=cfg.start_vel_max,
            tilt_max=cfg.start_tilt_max,
            ang_vel_max=cfg.start_ang_vel_max,
            rpm_jitter=cfg.hover_rpm_jitter,
        )
        # Domain randomisation: per-world mass / inertia (crazyflow params carry a world axis) and
        # a motor-gain factor standing in for thrust-curve error (rpm2thrust has no world axis).
        k_mass, k_inertia, k_gain = jax.random.split(k_dr, 3)
        mass = self.mass0 * (
            1.0
            + jax.random.uniform(k_mass, (n, 1, 1), minval=-params.dr_mass, maxval=params.dr_mass)
        )
        inertia_scale = 1.0 + jax.random.uniform(
            k_inertia, (n, 1, 3, 1), minval=-params.dr_inertia, maxval=params.dr_inertia
        )
        inertia = jnp.asarray(self.J0)[None, None] * inertia_scale
        motor_gain = 1.0 + jax.random.uniform(
            k_gain, (n,), minval=-params.dr_motor_gain, maxval=params.dr_motor_gain
        )

        sim = state.sim
        zeros3 = jnp.zeros((n, 1, 3))
        states = leaf_replace(
            sim.states,
            mask,
            pos=pos[:, None],
            quat=quat[:, None],
            vel=vel[:, None],
            ang_vel=ang_vel[:, None],
            rotor_vel=rotor_vel[:, None],
            force=zeros3,
            torque=zeros3,
        )
        hover_cmd = jnp.full((n, 1, 4), self.hover_rpm)
        controls = sim.controls.replace(
            rotor_vel=jnp.where(mask[:, None, None], hover_cmd, sim.controls.rotor_vel)
        )
        sim_params = leaf_replace(
            sim.params, mask, mass=mass, J=inertia, J_inv=jnp.linalg.inv(inertia)
        )
        core = sim.core.replace(steps=jnp.where(mask[:, None], 0, sim.core.steps))
        sim = sim.replace(states=states, controls=controls, params=sim_params, core=core)

        def sel(new: Array, old: Array) -> Array:
            m = mask.reshape((n,) + (1,) * (new.ndim - 1))
            return jnp.where(m, new, old)

        return state.replace(
            sim=sim,
            gates_pos=sel(gates_pos, state.gates_pos),
            gates_yaw=sel(gates_yaw, state.gates_yaw),
            n_active=sel(n_active, state.n_active),
            gate_idx=sel(jnp.zeros_like(state.gate_idx), state.gate_idx),
            prev_pos=sel(pos, state.prev_pos),
            prev_action=sel(jnp.zeros_like(state.prev_action), state.prev_action),
            t=sel(jnp.zeros_like(state.t), state.t),
            ep_return=sel(jnp.zeros_like(state.ep_return), state.ep_return),
            motor_gain=sel(motor_gain, state.motor_gain),
        )

    # ------------------------------------------------------------------------------ observation
    def observe(self, state: EnvState) -> Array:
        cfg, n = self.cfg, self.n_envs
        st = state.sim.states
        pos, quat, vel, ang_vel = st.pos[:, 0], st.quat[:, 0], st.vel[:, 0], st.ang_vel[:, 0]
        rotor = st.rotor_vel[:, 0]
        ar = jnp.arange(n)
        idx = state.gate_idx
        idx_next = jnp.minimum(idx + 1, self.n_total_gates - 1)
        g0, y0 = state.gates_pos[ar, idx], state.gates_yaw[ar, idx]
        g1, y1 = state.gates_pos[ar, idx_next], state.gates_yaw[ar, idx_next]
        n0, u0 = gate_frame(y0)
        n1, u1 = gate_frame(y1)
        to_body = lambda v: quat_rotate_inv(quat, v)
        gravity_dir = to_body(jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), (n, 3)))
        remaining = (state.n_active - idx).astype(jnp.float32)[:, None]
        rotor_norm = (rotor / self.hover_rpm - 1.0) / cfg.action_scale
        return jnp.concatenate(
            [
                pos[:, 2:3],
                gravity_dir,
                to_body(vel),
                ang_vel,
                rotor_norm,
                state.prev_action,
                to_body(g0 - pos),
                to_body(n0),
                to_body(u0),
                to_body(g1 - pos),
                to_body(n1),
                to_body(u1),
                remaining,
            ],
            axis=-1,
        )

    # ------------------------------------------------------------------------------------ step
    def step(
        self, state: EnvState, action: Array, params: EnvParams
    ) -> tuple[EnvState, Array, Array, Array, Array, Array, dict[str, Array]]:
        """Advance one control step (``n_ticks`` dynamics ticks) and auto-reset finished worlds.

        Returns:
            ``(state, obs, reward, terminated, truncated, final_obs, info)``. ``obs`` is the
            observation *after* auto-reset; ``final_obs`` is the pre-reset observation, needed to
            bootstrap the value of truncated episodes.
        """
        cfg, n = self.cfg, self.n_envs
        a = jnp.clip(action, -1.0, 1.0)
        rpm = state.motor_gain[:, None] * self.hover_rpm * (1.0 + cfg.action_scale * a)
        sim = F.rotor_vel_control(state.sim, rpm[:, None, :])
        sim = self._step_fn(sim, n_steps=self.n_ticks)

        st = sim.states
        pos, quat, vel, ang_vel = st.pos[:, 0], st.quat[:, 0], st.vel[:, 0], st.ang_vel[:, 0]
        ar = jnp.arange(n)
        idx = state.gate_idx
        g, yaw = state.gates_pos[ar, idx], state.gates_yaw[ar, idx]

        passed, hit, missed = gate_crossing(
            state.prev_pos, pos, g, yaw, cfg.half_opening, cfg.frame_thickness, cfg.drone_radius
        )
        d_prev = jnp.linalg.norm(state.prev_pos - g, axis=-1)
        d_new = jnp.linalg.norm(pos - g, axis=-1)
        progress = d_prev - d_new

        rel_b = quat_rotate_inv(quat, g - pos)
        cos_heading = rel_b[:, 0] / (jnp.linalg.norm(rel_b[:, :2], axis=-1) + 1e-6)
        r_yaw = -cfg.k_yaw * (1.0 - cos_heading)
        r_rate = -cfg.k_ang_vel * jnp.sum(ang_vel**2, axis=-1)
        r_act = -cfg.k_action_rate * jnp.sum((a - state.prev_action) ** 2, axis=-1)

        floor = pos[:, 2] < cfg.z_floor
        ceiling = pos[:, 2] > cfg.z_ceiling
        far = d_new > cfg.max_gate_dist
        crashed = hit | floor | ceiling | far
        gate_idx_new = idx + passed.astype(jnp.int32)
        success = passed & (gate_idx_new >= state.n_active)
        terminated = crashed | missed | success

        reward = (
            cfg.k_progress * progress
            + cfg.gate_bonus * passed
            - cfg.crash_penalty * crashed
            - cfg.miss_penalty * missed
            + r_yaw
            + r_rate
            + r_act
        )
        t_new = state.t + 1
        max_steps = cfg.max_steps_base + cfg.steps_per_gate * state.n_active
        truncated = (t_new >= max_steps) & ~terminated
        done = terminated | truncated
        ep_return = state.ep_return + reward

        new_state = state.replace(
            sim=sim,
            gate_idx=jnp.minimum(gate_idx_new, self.n_total_gates - 1),
            prev_pos=pos,
            prev_action=a,
            t=t_new,
            ep_return=ep_return,
        )
        final_obs = self.observe(new_state)
        info = {
            "episode_return": ep_return,
            "episode_length": t_new.astype(jnp.float32),
            "gates_passed": gate_idx_new.astype(jnp.float32),
            "success": success.astype(jnp.float32),
            "crashed": crashed.astype(jnp.float32),
            "missed": missed.astype(jnp.float32),
            "speed": jnp.linalg.norm(vel, axis=-1),
            "rpm_cmd": rpm,
            "pos": pos,
            "quat": quat,
            "vel": vel,
        }
        rng, key = jax.random.split(state.rng)
        reset_state = self._reset_worlds(new_state.replace(rng=rng), done, key, params)
        return (
            reset_state,
            self.observe(reset_state),
            reward,
            terminated,
            truncated,
            final_obs,
            info,
        )
