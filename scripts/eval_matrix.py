"""Run the evaluation matrix behind the report figures for one checkpoint, in one process.

Usage (from the repo root):

    uv run python scripts/eval_matrix.py --stage mixed_tracks \
        --ckpt outputs/mixed_tracks/<run>/checkpoint.pkl --out outputs/mixed_tracks/<run>

Writes one ``gate-ppo-eval --json`` file per cell, named as ``scripts/make_report_figures.py``
expects. The settings are those of the 2026-09-08 results (seed 0, 1,024 episodes, deterministic,
DR on unless stated). Note that the ``mixed_tracks`` "chain" cells pass no ``--track-type`` and so
use the checkpoint's own training distribution (half chains with gate normals rotated by up to
+-0.3 rad, half circles with 5-12 gates per turn); ``plainchain*`` cells are plain chains.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from gate_ppo.evaluate import main as evaluate  # noqa: E402


def circles(suffix: str = "") -> dict[str, list[str]]:
    cells = {
        f"eval_circle_gpt{g}{suffix}": ["--n-gates", "5", "--track-type", "circle"]
        + ["--gates-per-turn", str(g)]
        for g in (12, 8, 6, 5, 4)
    }
    full = ["--track-type", "circle", "--gates-per-turn"]
    cells[f"eval_circle_gpt10_full{suffix}"] = ["--n-gates", "10", *full, "10"]
    return cells


CHAIN = ["--track-type", "chain"]
MATRIX: dict[str, dict[str, list[str]]] = {
    "three_gates": {
        "eval_3gates": ["--n-gates", "3"],
        "eval_5gates": ["--n-gates", "5"],
        "eval_5gates_nodr": ["--n-gates", "5", "--no-dr"],
        "eval_5gates_stoch": ["--n-gates", "5", "--stochastic"],
        "eval_7gates": ["--n-gates", "7"],
        "eval_10gates": ["--n-gates", "10"],
        **circles(),
        "eval_circle_gpt8_full": ["--n-gates", "8", "--track-type", "circle"]
        + ["--gates-per-turn", "8"],
    },
    "five_gates": {
        "eval_3gates_det": ["--n-gates", "3"],
        "eval_5gates_det": ["--n-gates", "5"],
        "eval_5gates_det_nodr": ["--n-gates", "5", "--no-dr"],
        "eval_5gates_stoch": ["--n-gates", "5", "--stochastic"],
        "eval_7gates_det_capped": ["--n-gates", "7"],
        "eval_10gates_det_capped": ["--n-gates", "10"],
        **circles("_zeroshot"),
        # the heatmap's "full circle, 8 gates" cell of this row used the 256-world trajectory run
        "eval_circle8_traj": ["--n-gates", "8", "--track-type", "circle"]
        + ["--gates-per-turn", "8", "--n-episodes", "256"],
        "eval_circle_gpt10_zeroshot": ["--n-gates", "5", "--track-type", "circle"]
        + ["--gates-per-turn", "10"],
        "eval_circle_gpt24_zeroshot": ["--n-gates", "5", "--track-type", "circle"]
        + ["--gates-per-turn", "24"],
        "eval_circle_gpt36_zeroshot": ["--n-gates", "5", "--track-type", "circle"]
        + ["--gates-per-turn", "36"],
        **{
            f"eval_chain_gyn{tag}_zeroshot": ["--n-gates", "5", *CHAIN, "--gate-yaw-noise", gyn]
            for tag, gyn in (("013", "0.13"), ("026", "0.26"), ("052", "0.52"))
        },
    },
    "mixed_tracks": {
        "eval_chain3": ["--n-gates", "3"],
        "eval_chain5": ["--n-gates", "5"],
        "eval_chain5_nodr": ["--n-gates", "5", "--no-dr"],
        "eval_chain5_stoch": ["--n-gates", "5", "--stochastic"],
        "eval_chain7": ["--n-gates", "7"],
        "eval_chain10": ["--n-gates", "10"],
        "eval_chain5_gyn03": ["--n-gates", "5", "--gate-yaw-noise", "0.3"],
        **circles(),
        "eval_circle_gpt8_full": ["--n-gates", "8", "--track-type", "circle"]
        + ["--gates-per-turn", "8"],
        "eval_plainchain5": ["--n-gates", "5", *CHAIN, "--gate-yaw-noise", "0"],
        "eval_plainchain10": ["--n-gates", "10", *CHAIN, "--gate-yaw-noise", "0"],
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", required=True, choices=sorted(MATRIX))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*", default=None, help="subset of cell names")
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cell in MATRIX[args.stage].items():
        if args.only and name not in args.only:
            continue
        path = args.out / f"{name}.json"
        argv = ["--ckpt", args.ckpt, "--json", str(path), "--device", args.device, *cell]
        summary = evaluate(argv)
        rows.append((name, summary["success_rate"]))
    print(json.dumps(dict(rows), indent=2))


if __name__ == "__main__":
    main()
