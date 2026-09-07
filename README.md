# crazyflow-gate-ppo

End-to-end JAX PPO expert that flies a Crazyflie (`cf2x_L250`) through chains of gates in the
[crazyflow](https://github.com/learnsyslab/crazyflow) simulator. This is **Phase 1 of UAVLA**: the
RL expert is the *data generator* for the later SmolVLA + normalizing-flow policy, not the deliverable
itself. Design decisions follow the project's research notes.

- **Actions:** 4 motor RPMs at 48 Hz (`rpm = hover_rpm · (1 + 0.35·a)`, `a ∈ [-1, 1]⁴`) over 480 Hz
  first-principles dynamics — the commanded RPM is the label the VLA will be trained on.
- **Task:** randomised chains of 3–5 upright square gates (0.49 m opening, ±0.6 rad turns, 1.2–2.2 m
  spacing, altitude 0.7–2.2 m). Gate passage / frame collision / miss are geometric tests in JAX
  (no MJX contact queries); worlds auto-reset in the same step.
- **Learner:** PureJaxRL-style PPO — physics, gate logic, policy, GAE and the clipped update live in
  one `jax.jit` per iteration. 2×256 tanh actor and critic, truncation bootstrapping, running
  observation normalisation, entropy bonus + linear LR annealing, per-world domain randomisation
  (mass, inertia, motor gain), yaw-range curriculum.

## Setup

```bash
uv sync                      # Python 3.12, crazyflow[gpu]==0.3.2 (CUDA 12), flax, optax, hydra
uv run pytest                # CPU test suite (~30 s)
uv run ruff check src tests
```

`train.py` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX can share the GPU with PyTorch later;
`gate_ppo/__init__.py` sets `SCIPY_ARRAY_API=1`, which crazyflow requires before scipy is imported.

## Train

```bash
# stage 1: randomised 2-3 gate tracks, 30 M steps (~1.2 min on an RTX 2080)
uv run gate-ppo-train experiment=three_gates

# stage 2: randomised 3-5 gate tracks, 50 M steps, warm-started from stage 1 (~2 min)
uv run gate-ppo-train experiment=five_gates run.init_from=outputs/three_gates/<run>/checkpoint.pkl

# anything is a Hydra override; multirun over seeds:
uv run gate-ppo-train experiment=five_gates run.seed=0,1,2 -m
uv run gate-ppo-train run.device=cpu env.n_envs=256      # laptop CPU / Mac
```

Outputs go to `outputs/<run.name>/<timestamp>/` (`checkpoint.pkl`, `metrics.jsonl`, the resolved
config under `.hydra/`). Console lines show episode return/length, success, gates passed, crash / miss
/ timeout rates, mean speed, entropy, KL, `log_std` and the current curriculum yaw range. Set
`run.wandb=true` (after `uv sync --extra wandb`) to log to Weights & Biases.

## Evaluate and render

```bash
uv run gate-ppo-eval --ckpt outputs/five_gates/<run>/checkpoint.pkl --n-gates 5 --n-episodes 1024
uv run gate-ppo-eval --ckpt ... --n-gates 10                 # longer chains than seen in training
uv run gate-ppo-eval --ckpt ... --n-gates 5 --stochastic     # sample actions (demo diversity)
uv run gate-ppo-eval --ckpt ... --n-gates 5 --save-traj traj.npz --traj-worlds 16
uv run gate-ppo-render --traj traj.npz --world 0 --camera track --out flight.mp4   # or --camera fpv
```

Evaluation runs one fresh episode per world (deterministic mean action by default) and reports success
/ crash / miss / timeout rates and gates passed. `gate-ppo-render` replays logged poses through
crazyflow's MuJoCo renderer with the gates added to the scene — the same second-pass pattern Phase 2
will use to produce FPV frames for the demonstration dataset.

## Results (2026-09-08, seed 0, laptop RTX 2080 8 GB)

Throughput: **~4.5–5.2 × 10⁵ env steps/s** at 1024 worlds × 32 steps per update (measured; the
vault's estimate for this GPU was a tier-3 guess of 2 × 10⁵–10⁶). Compilation ~8 s.

| Stage | Steps | Wall time | Train success (stochastic, last iters) |
|---|---|---|---|
| `three_gates` (2–3 gates, curriculum 0.2 → 0.6 rad) | 30 M | 1.2 min | 0.94–0.98 |
| `five_gates` (3–5 gates, warm-started) | 50 M | 2.1 min | 0.94–0.98 |

Evaluation of the `five_gates` checkpoint, 1024 fresh tracks each, all gates required:

| Track | Policy | DR | Success | Crash | Miss | Gates passed | Time |
|---|---|---|---|---|---|---|---|
| 5 gates | deterministic | on | **0.957** | 0.042 | 0.001 | 4.90 / 5 | 2.0 s |
| 5 gates | deterministic | off | 0.984 | 0.016 | 0 | – | – |
| 5 gates | stochastic | on | 0.941 | 0.059 | 0 | 4.87 / 5 | 2.0 s |
| 7 gates | deterministic | on | 0.876 | 0.123 | 0.001 | 6.57 / 7 | 2.5 s |
| 10 gates | deterministic | on | 0.765 | 0.232 | 0.003 | 8.74 / 10 | 3.2 s |
| 7 gates, uncapped `remaining` feature | deterministic | on | 0.835 | 0.163 | 0.002 | 6.37 / 7 | 2.5 s |
| 10 gates, uncapped `remaining` feature | deterministic | on | 0.392 | 0.597 | 0.012 | 5.73 / 10 | 2.3 s |

The `three_gates` checkpoint already reaches 0.908 on 5-gate chains zero-shot (0.937 on 3-gate
chains), i.e. the gate-count curriculum is nearly free — the policy is a local track follower that
only sees the current and next gate.

On chains longer than the training tracks, the first version of the observation exposed the raw
"gates remaining" count (6–10, never seen by the running normaliser) and success on 10-gate chains
collapsed to 0.39 with crashes at the *first* gates — not a speed problem (speed plateaus at
≈ 5.8 m/s by gate 3 and then decreases). Clipping that feature at the training horizon
(`env.obs_remaining_cap`, default 5) restores 0.77 on 10-gate chains with the same checkpoint; the
rest of the gap is per-gate crash risk compounding over more gates. Training on longer chains and
more seeds are the obvious next experiments.

## Layout

```
configs/            Hydra config + experiment overrides (single_gate, three_gates, five_gates)
src/gate_ppo/
  config.py         dataclass schema (defaults for every knob)
  geometry.py       quaternion helpers, gate frames, the passage / frame / miss test, hover RPM
  track.py          randomised gate chains and initial states
  env.py            GateRaceEnv: EnvState/EnvParams pytrees, reset, observe, step (autoreset)
  networks.py       actor / critic MLPs
  ppo.py            rollout + GAE + clipped PPO update in one jit, running obs stats, metrics
  train.py          Hydra entry point, curriculum, logging, checkpoints
  evaluate.py       first-episode outcomes on fresh chains, trajectory dump
  render.py         replay a trajectory to MP4 with gates in the MuJoCo scene
tests/              CPU tests: geometry, env invariants (hover holds altitude), one PPO iteration
```

## Gotchas learned the hard way

- **Rotors must be pre-spun at reset, and equally.** crazyflow's rotors start at 0 RPM and follow
  first-order dynamics (τ ≈ 0.14 s). A ±5 % per-motor jitter at reset is an undamped torque impulse
  that leaves the drone tumbling at ~6 rad/s; use `env.start_ang_vel_max` for that kind of
  diversity instead (`hover_rpm_jitter` defaults to 0).
- **crazyflow 0.3.2 does not clip rotor speeds** (no `clip_rotor_vel` stage in the step pipeline,
  unlike the current docs); the action mapping bounds RPM to `[0.65, 1.35] · hover`.
- Only per-world parameters (`mass`, `J`) can be randomised per world; `rpm2thrust` and
  `rotor_dyn_coef` have no world axis, hence the motor-gain factor in the action mapping.
- `sim.contacts()` costs ~5× on CPU — gate events are geometric on purpose.
