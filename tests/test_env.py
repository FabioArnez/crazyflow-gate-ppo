import jax
import jax.numpy as jnp
import numpy as np

from gate_ppo.config import EnvConfig
from gate_ppo.env import OBS_DIM, EnvParams, GateRaceEnv


def make_env(n_envs=4, n_gates=3, n_gates_min=2):
    cfg = EnvConfig(n_envs=n_envs, n_gates=n_gates, n_gates_min=n_gates_min)
    env = GateRaceEnv(cfg, device="cpu")
    params = EnvParams.from_config(cfg)
    state = env.init_state(jax.random.key(0))
    state, obs = env.reset(state, params)
    return env, params, state, obs


def test_reset_shapes_and_track_constraints():
    env, params, state, obs = make_env()
    cfg = env.cfg
    assert obs.shape == (4, OBS_DIM)
    assert bool(jnp.all(jnp.isfinite(obs)))
    gates = np.asarray(state.gates_pos)
    assert gates.shape == (4, cfg.n_gates + 1, 3)
    assert np.all(gates[..., 2] >= cfg.z_min - 1e-6) and np.all(gates[..., 2] <= cfg.z_max + 1e-6)
    spacing = np.linalg.norm(np.diff(gates[:, :, :2], axis=1), axis=-1)
    assert np.all(spacing >= cfg.spacing_min - 1e-4) and np.all(spacing <= cfg.spacing_max + 1e-4)
    n_active = np.asarray(state.n_active)
    assert np.all(n_active >= 2) and np.all(n_active <= 3)
    # Rotors are pre-spun to hover and the drone's params were randomised per world.
    rotor = np.asarray(state.sim.states.rotor_vel[:, 0])
    assert np.all(np.abs(rotor / env.hover_rpm - 1.0) <= cfg.hover_rpm_jitter + 1e-6)
    mass = np.asarray(state.sim.params.mass[:, 0, 0])
    assert np.all(np.abs(mass / env.mass0 - 1.0) <= cfg.dr_mass + 1e-6)


def test_hover_action_holds_altitude():
    env, params, state, obs = make_env()
    params = params.replace(dr_mass=jnp.float32(0.0), dr_motor_gain=jnp.float32(0.0))
    state, obs = env.reset(state, params)
    z0 = np.asarray(state.sim.states.pos[:, 0, 2])
    step = jax.jit(env.step)
    for _ in range(env.cfg.ctrl_freq):  # one second at the hover command
        state, obs, reward, terminated, truncated, final_obs, info = step(
            state, jnp.zeros((4, 4)), params
        )
        assert not bool(jnp.any(terminated | truncated))
    z1 = np.asarray(state.sim.states.pos[:, 0, 2])
    # initial velocity is random (+-0.3 m/s) and decays through drag; altitude stays close.
    assert np.all(np.abs(z1 - z0) < 0.4), (z0, z1)


def test_step_outputs_and_autoreset():
    env, params, state, obs = make_env()
    step = jax.jit(env.step)
    key = jax.random.key(1)
    for _ in range(200):
        key, k = jax.random.split(key)
        action = jax.random.uniform(k, (4, 4), minval=-1.0, maxval=1.0)
        state, obs, reward, terminated, truncated, final_obs, info = step(state, action, params)
        assert obs.shape == final_obs.shape == (4, OBS_DIM)
        assert bool(jnp.all(jnp.isfinite(obs))) and bool(jnp.all(jnp.isfinite(reward)))
        done = np.asarray(terminated | truncated)
        # Worlds that finished were re-initialised in the same step.
        assert np.all(np.asarray(state.t)[done] == 0)
        assert np.all(np.asarray(state.gate_idx)[done] == 0)
    # Random RPM commands crash quickly, so at least one world must have reset by now.
    assert done.any() or np.asarray(state.t).max() < 200


def test_rotor_command_clipped_to_motor_limits():
    env, params, state, obs = make_env()
    # crazyflow >= 0.4 clips at thrust_max; full throttle (1.35 x hover) lies above it.
    assert env.rpm_min < env.hover_rpm < env.rpm_max < 1.35 * env.hover_rpm
    assert state.sim.params.mass.shape == (4, 1, 1) and state.sim.params.J.shape == (4, 1, 3, 3)
    *_, info = jax.jit(env.step)(state, jnp.ones((4, 4)), params)
    np.testing.assert_allclose(np.asarray(info["rpm_cmd"]), env.rpm_max, rtol=1e-6)
