"""Typed configuration schema (registered with Hydra's ConfigStore in ``train.py``).

Values here are the defaults; ``configs/config.yaml`` and ``configs/experiment/*.yaml`` override
them. Anything that changes *during* training (curriculum, domain randomization ranges) is mirrored
into ``env.EnvParams`` so it can be traced by JAX without recompiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # --- simulator -------------------------------------------------------------------------
    drone: str = "cf2x_L250"
    sim_freq: int = 480  # dynamics ticks / s
    ctrl_freq: int = 48  # policy rate; sim_freq // ctrl_freq ticks per control step (10)
    n_envs: int = 1024  # crazyflow worlds stepped in one fused kernel
    # --- actions: rpm = motor_gain * hover_rpm * (1 + action_scale * clip(a, -1, 1)) -------------
    action_scale: float = 0.35
    # --- track sampling (a chain of n_gates + 1 upright square gates) ---------------------------
    n_gates: int = 5  # max active gates per episode; sizes the gate arrays (static)
    n_gates_min: int = 3  # active gates per world ~ U{n_gates_min, ..., n_gates}
    spacing_min: float = 1.2  # m between consecutive gates
    spacing_max: float = 2.2
    yaw_range: float = 0.6  # max |heading change| between consecutive gates (rad)
    dz_max: float = 0.3  # max altitude change between consecutive gates (m)
    z_min: float = 0.7  # gate altitude clipped to [z_min, z_max]
    z_max: float = 2.2
    start_z_min: float = 0.8
    start_z_max: float = 1.4
    start_xy_jitter: float = 0.2
    start_yaw_noise: float = 0.3
    start_vel_max: float = 0.3
    start_tilt_max: float = 0.1
    start_ang_vel_max: float = 0.2  # rad/s
    # Rotors are pre-spun to hover * (1 +- jitter) at reset. Keep this small: unequal rotor speeds
    # during the ~0.14 s spin-up act as an undamped torque impulse (5 % -> ~6 rad/s of tumble).
    hover_rpm_jitter: float = 0.0
    # --- gate geometry (drone-data-collect gate_14.urdf; cf2x_L250 collision sphere) -----------
    half_opening: float = 0.245  # inner opening is 0.49 m square
    frame_thickness: float = 0.06
    drone_radius: float = 0.07
    # --- termination bounds --------------------------------------------------------------------
    z_floor: float = 0.05
    z_ceiling: float = 3.5
    max_gate_dist: float = 6.0
    # --- episode length: max_steps = max_steps_base + steps_per_gate * n_active ----------------
    max_steps_base: int = 96
    steps_per_gate: int = 96
    # The "gates remaining" observation is clipped here so that chains longer than the training
    # tracks stay in-distribution (the policy is a local track follower; it only needs to know
    # when the last gate is near). Keep it at the largest n_gates used for training.
    obs_remaining_cap: int = 5
    # --- reward (Song et al., 2021 style) ------------------------------------------------------
    k_progress: float = 1.0  # per metre of progress towards the current gate
    gate_bonus: float = 5.0
    crash_penalty: float = 10.0  # floor / ceiling / frame hit / out of bounds
    miss_penalty: float = 10.0  # crossed the gate plane outside the frame
    k_ang_vel: float = 2e-3  # * |omega|^2
    k_action_rate: float = 1e-2  # * |a_t - a_{t-1}|^2
    k_yaw: float = (
        1e-2  # * (1 - cos(heading error to current gate)); keeps the FPV camera on the gate
    )
    # --- domain randomization (+- fraction, uniform, per world, resampled at every reset) ------
    dr_mass: float = 0.1
    dr_inertia: float = 0.1
    dr_motor_gain: float = 0.05


@dataclass
class NetConfig:
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    log_std_init: float = -0.5


@dataclass
class PPOConfig:
    n_steps: int = 32  # rollout length per world; batch = n_envs * n_steps
    n_epochs: int = 4
    n_minibatches: int = 8
    lr: float = 3e-4
    anneal_lr: bool = True  # linear decay to 0 over the run (mitigates entropy collapse)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    clip_vloss: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 1e-3
    max_grad_norm: float = 0.5
    normalize_obs: bool = True  # running Welford statistics carried in the train state
    normalize_adv: bool = True
    total_env_steps: int = 30_000_000


@dataclass
class CurriculumConfig:
    """Grow the gate-yaw range once the recent success rate clears a threshold."""

    enabled: bool = True
    yaw_range_init: float = 0.2
    yaw_range_step: float = 0.1
    success_threshold: float = 0.8
    window: int = 10  # iterations averaged for the decision


@dataclass
class RunConfig:
    name: str = "gate_ppo"
    seed: int = 0
    device: str = "gpu"  # "gpu" | "cpu"
    log_interval: int = 10  # iterations between console lines (metrics.jsonl gets every iteration)
    ckpt_interval: int = 100
    init_from: str | None = None  # checkpoint .pkl to warm-start params + obs stats from
    wandb: bool = False
    wandb_project: str = "uavla-gate-ppo"
    jax_cache: bool = True  # persistent XLA compilation cache (crazyflow.utils.enable_cache)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    net: NetConfig = field(default_factory=NetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    run: RunConfig = field(default_factory=RunConfig)
