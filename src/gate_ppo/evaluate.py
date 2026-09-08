"""Evaluate a checkpoint on fresh gate chains.

Example: ``gate-ppo-eval --ckpt outputs/five_gates/<run>/checkpoint.pkl --n-gates 7``.

Every world runs one episode from a fresh reset; the outcome of that first episode is recorded
(the environment auto-resets afterwards, which is ignored). ``--save-traj`` dumps the first
``--traj-worlds`` worlds' trajectories for ``gate-ppo-render``.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from gate_ppo.checkpoint import load_checkpoint  # noqa: E402
from gate_ppo.config import EnvConfig  # noqa: E402
from gate_ppo.env import EnvParams, GateRaceEnv  # noqa: E402
from gate_ppo.networks import ActorCritic  # noqa: E402
from gate_ppo.ppo import RunningStats  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ckpt", required=True, help="checkpoint.pkl written by gate-ppo-train")
    p.add_argument(
        "--n-gates", type=int, default=None, help="gates per track (default: training value)"
    )
    p.add_argument("--n-gates-min", type=int, default=None, help="default: same as --n-gates")
    p.add_argument("--n-episodes", type=int, default=1024, help="worlds = episodes")
    p.add_argument("--yaw-range", type=float, default=None, help="default: training env.yaw_range")
    p.add_argument("--track-type", choices=["chain", "circle", "mixed"], default=None)
    p.add_argument("--gates-per-turn", type=float, default=None, help="circle: turn = 2*pi/this")
    p.add_argument("--circle-prob", type=float, default=None, help="mixed: share of circles")
    p.add_argument("--gate-yaw-noise", type=float, default=None, help="chain: normal offset (rad)")
    p.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    p.add_argument("--no-dr", action="store_true", help="disable domain randomisation")
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-traj", type=str, default=None, help="write trajectories to this .npz")
    p.add_argument("--traj-worlds", type=int, default=8)
    p.add_argument("--json", type=str, default=None, help="write the summary to this .json")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    ckpt = load_checkpoint(args.ckpt)
    cfg = ckpt["config"]
    n_gates = args.n_gates or cfg["env"]["n_gates"]
    n_gates_min = args.n_gates_min or n_gates
    overrides = {"n_envs": args.n_episodes, "n_gates": n_gates, "n_gates_min": n_gates_min}
    if args.track_type:
        overrides["track_type"] = args.track_type
    if args.gate_yaw_noise is not None:
        overrides["gate_yaw_noise"] = args.gate_yaw_noise
    env_cfg = EnvConfig(**{**cfg["env"], **overrides})
    env = GateRaceEnv(env_cfg, device=args.device)
    network = ActorCritic(env.action_dim, tuple(cfg["net"]["hidden"]), cfg["net"]["log_std_init"])
    params = jax.tree.map(jnp.asarray, ckpt["params"])
    stats = RunningStats(
        mean=jnp.asarray(ckpt["obs_stats"]["mean"]),
        var=jnp.asarray(ckpt["obs_stats"]["var"]),
        count=jnp.float32(ckpt["obs_stats"]["count"]),
    )
    normalize_obs = cfg["ppo"]["normalize_obs"]
    env_params = EnvParams.from_config(
        env_cfg,
        yaw_range=args.yaw_range,
        domain_randomization=not args.no_dr,
        circle_prob=args.circle_prob,
        gates_per_turn=args.gates_per_turn,
    )
    max_steps = env_cfg.max_steps_base + env_cfg.steps_per_gate * n_gates
    k_env, k_run = jax.random.split(jax.random.key(args.seed))
    state = env.init_state(k_env)
    state, obs = env.reset(state, env_params)
    n_traj = min(args.traj_worlds, args.n_episodes)
    track = {
        "gates_pos": np.asarray(state.gates_pos[:n_traj]),
        "gates_yaw": np.asarray(state.gates_yaw[:n_traj]),
        "n_active": np.asarray(state.n_active[:n_traj]),
    }

    def policy(obs, key):
        mean, log_std, _ = network.apply(params, stats.normalize(obs) if normalize_obs else obs)
        if args.stochastic:
            return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        return mean

    @jax.jit
    def rollout(state, obs, key):
        n = env.n_envs
        outcome0 = {
            k: jnp.zeros((n,))
            for k in ("success", "crashed", "missed", "timeout", "gates", "length", "return")
        }

        def body(carry, _):
            state, obs, key, first_done, outcome = carry
            key, k = jax.random.split(key)
            action = policy(obs, k)
            state, obs, reward, terminated, truncated, final_obs, info = env.step(
                state, action, env_params
            )
            done = terminated | truncated
            new = done & ~first_done
            values = {
                "success": info["success"],
                "crashed": info["crashed"],
                "missed": info["missed"],
                "timeout": truncated.astype(jnp.float32),
                "gates": info["gates_passed"],
                "length": info["episode_length"],
                "return": info["episode_return"],
            }
            outcome = {k: jnp.where(new, values[k], outcome[k]) for k in outcome}
            traj = {
                "pos": info["pos"][:n_traj],
                "quat": info["quat"][:n_traj],
                "vel": info["vel"][:n_traj],
                "rpm_cmd": info["rpm_cmd"][:n_traj],
                "gates_passed": info["gates_passed"][:n_traj],
                "done": done[:n_traj],
            }
            return (state, obs, key, first_done | done, outcome), traj

        carry, traj = jax.lax.scan(
            body, (state, obs, key, jnp.zeros((n,), bool), outcome0), None, max_steps
        )
        return carry[3], carry[4], traj

    finished, outcome, traj = jax.device_get(rollout(state, obs, k_run))
    finished = np.asarray(finished)
    # Worlds still flying at max_steps count as timeouts.
    for k in outcome:
        outcome[k] = np.asarray(outcome[k])
    outcome["timeout"] = np.where(finished, outcome["timeout"], 1.0)
    n_active = np.asarray(state.n_active)
    summary = {
        "checkpoint": str(args.ckpt),
        "n_gates": n_gates,
        "n_gates_min": n_gates_min,
        "n_episodes": args.n_episodes,
        "yaw_range": float(env_params.yaw_range),
        "track_type": env_cfg.track_type,
        "gate_yaw_noise": env_cfg.gate_yaw_noise,
        "gates_per_turn": [
            float(env_params.circle_gates_per_turn_min),
            float(env_params.circle_gates_per_turn_max),
        ],
        "deterministic": not args.stochastic,
        "domain_randomization": not args.no_dr,
        "success_rate": float(outcome["success"].mean()),
        "crash_rate": float(outcome["crashed"].mean()),
        "miss_rate": float(outcome["missed"].mean()),
        "timeout_rate": float(outcome["timeout"].mean()),
        "gates_passed_mean": float(outcome["gates"].mean()),
        "episode_length_s": float(outcome["length"].mean() / env_cfg.ctrl_freq),
        "return_mean": float(outcome["return"].mean()),
        "per_n_active": {
            int(k): float(outcome["success"][n_active == k].mean()) for k in np.unique(n_active)
        },
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    if args.save_traj:
        done = np.asarray(traj["done"])  # (T, K)
        ep_len = np.where(done.any(0), done.argmax(0) + 1, done.shape[0])
        np.savez(
            args.save_traj,
            pos=np.asarray(traj["pos"]),
            quat=np.asarray(traj["quat"]),
            vel=np.asarray(traj["vel"]),
            rpm_cmd=np.asarray(traj["rpm_cmd"]),
            gates_passed=np.asarray(traj["gates_passed"]),
            ep_len=ep_len,
            success=outcome["success"][:n_traj],
            ctrl_freq=env_cfg.ctrl_freq,
            drone=env_cfg.drone,
            sim_freq=env_cfg.sim_freq,
            half_opening=env_cfg.half_opening,
            frame_thickness=env_cfg.frame_thickness,
            **track,
        )
        print(f"trajectories of {n_traj} worlds -> {args.save_traj}")
    return summary


if __name__ == "__main__":
    main()
