"""Build the figures of the Phase-1 implementation report from run outputs.

Usage (from the repo root, after the three training stages and the evaluations):

    uv run python scripts/make_report_figures.py --out results/figures \
        --three outputs/three_gates/<run> --five outputs/five_gates/<run> --mixed outputs/mixed_tracks/<run>

Every figure is a plain PNG; the palette is the validated default of the data-viz method
(categorical slots 1-3, the sequential blue ramp, chrome/ink tokens), marks are thin, gridlines are
solid hairlines, one axis per panel.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

# ------------------------------------------------------------------ palette (documented instance)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (all-pairs safe)
SEQ = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]  # blue 100 -> 700
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
PREFIX = "phase1"  # figure file prefix, set by --prefix
TITLE_NOTE = ""  # second line of the heatmap title, set by --title-note
GATE_RED = "#e34948"  # slot 8, used only for gate frames in the map views (matches the renders)
DEEMPH = "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlelocation": "left",
    }
)


def read_jsonl(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = rows[0].keys()
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def success(path: Path) -> float:
    d = read_json(path)
    return float(d["success_rate"]) if d else np.nan


def rolling(x: np.ndarray, w: int = 10) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(len(x)):
        seg = x[max(0, i - w + 1) : i + 1]
        seg = seg[~np.isnan(seg)]
        out[i] = seg.mean() if len(seg) else np.nan
    return out


def rounded_bar(ax, x, height, width, color, radius=None, zorder=2):
    """Bar with a rounded data end and a square baseline (the mark spec), drawn as a path."""
    if height <= 0:
        return
    r = min(width * 0.25 if radius is None else radius, height / 2)
    x0, x1, y0, y1 = x - width / 2, x + width / 2, 0.0, height
    verts = [
        (x0, y0),
        (x1, y0),
        (x1, y1 - r),
        (x1, y1),
        (x1 - r, y1),
        (x0 + r, y1),
        (x0, y1),
        (x0, y1 - r),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def stage_label(ax, x, text):
    ax.axvline(x, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(
        x,
        1.0,
        " " + text,
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK2,
    )


# ------------------------------------------------------------------ figure 1: training curves
def fig_training(runs: dict[str, Path], out: Path):
    stages = [
        ("three_gates", runs["three"]),
        ("five_gates", runs["five"]),
        ("mixed_tracks", runs["mixed"]),
    ]
    data, offset, bounds = [], 0.0, []
    for name, d in stages:
        m = read_jsonl(d / "metrics.jsonl")
        m["x"] = (m["env_steps"] + offset) / 1e6
        bounds.append((offset / 1e6, name))
        offset += m["env_steps"][-1]
        data.append(m)
    panels = [
        ("episode/success", "Success rate", (0, 1.02)),
        ("episode/crash", "Crash and miss rate", (0, 1.02)),
        ("episode/gates_passed", "Gates passed per episode", None),
        ("rollout/speed", "Mean speed (m/s)", None),
        ("policy/log_std", "Policy log σ", None),
        ("curriculum/yaw_range", "Curriculum yaw range (rad)", (0, 0.7)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), constrained_layout=True)
    for ax, (key, title, ylim) in zip(axes.ravel(), panels, strict=True):
        for m in data:
            ax.plot(m["x"], m[key], color=SERIES[0], alpha=0.18, linewidth=0.8)
            ax.plot(m["x"], rolling(m[key]), color=SERIES[0])
            if key == "episode/crash":
                ax.plot(m["x"], m["episode/miss"], color=SERIES[1], alpha=0.18, linewidth=0.8)
                ax.plot(m["x"], rolling(m["episode/miss"]), color=SERIES[1])
        for xb, name in bounds:
            stage_label(ax, xb, name)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(0, offset / 1e6)
        if key == "episode/crash":
            ax.legend(
                handles=[
                    plt.Line2D([], [], color=SERIES[0], label="crash"),
                    plt.Line2D([], [], color=SERIES[1], label="miss"),
                ],
                loc="upper right",
            )
    for ax in axes[1]:
        ax.set_xlabel("environment steps (millions, stages concatenated)")
    fig.suptitle(
        "Training: three stages, 110 M steps, 4.7 min on an RTX 2080 (10-iteration rolling mean over raw)",
        fontsize=10,
        color=INK2,
        ha="left",
        x=0.01,
    )
    fig.savefig(out / f"{PREFIX}_fig1_training.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2: evaluation heatmap
def fig_heatmap(runs: dict[str, Path], out: Path):
    d3, d5, dm = runs["three"], runs["five"], runs["mixed"]
    cols = [
        (
            "chain\n3 gates",
            d3 / "eval_3gates.json",
            d5 / "eval_3gates_det.json",
            dm / "eval_chain3.json",
        ),
        (
            "chain\n5 gates",
            d3 / "eval_5gates.json",
            d5 / "eval_5gates_det.json",
            dm / "eval_chain5.json",
        ),
        (
            "chain 5\nno DR",
            d3 / "eval_5gates_nodr.json",
            d5 / "eval_5gates_det_nodr.json",
            dm / "eval_chain5_nodr.json",
        ),
        (
            "chain 5\nstochastic",
            d3 / "eval_5gates_stoch.json",
            d5 / "eval_5gates_stoch.json",
            dm / "eval_chain5_stoch.json",
        ),
        (
            "chain\n7 gates",
            d3 / "eval_7gates.json",
            d5 / "eval_7gates_det_capped.json",
            dm / "eval_chain7.json",
        ),
        (
            "chain\n10 gates",
            d3 / "eval_10gates.json",
            d5 / "eval_10gates_det_capped.json",
            dm / "eval_chain10.json",
        ),
        (
            "circle\n30°/gate",
            d3 / "eval_circle_gpt12.json",
            d5 / "eval_circle_gpt12_zeroshot.json",
            dm / "eval_circle_gpt12.json",
        ),
        (
            "circle\n45°/gate",
            d3 / "eval_circle_gpt8.json",
            d5 / "eval_circle_gpt8_zeroshot.json",
            dm / "eval_circle_gpt8.json",
        ),
        (
            "circle\n60°/gate",
            d3 / "eval_circle_gpt6.json",
            d5 / "eval_circle_gpt6_zeroshot.json",
            dm / "eval_circle_gpt6.json",
        ),
        (
            "circle\n72°/gate",
            d3 / "eval_circle_gpt5.json",
            d5 / "eval_circle_gpt5_zeroshot.json",
            dm / "eval_circle_gpt5.json",
        ),
        (
            "circle\n90°/gate",
            d3 / "eval_circle_gpt4.json",
            d5 / "eval_circle_gpt4_zeroshot.json",
            dm / "eval_circle_gpt4.json",
        ),
        (
            "full circle\n8 gates",
            d3 / "eval_circle_gpt8_full.json",
            d5 / "eval_circle8_traj.json",
            dm / "eval_circle_gpt8_full.json",
        ),
        (
            "full circle\n10 gates",
            d3 / "eval_circle_gpt10_full.json",
            d5 / "eval_circle_gpt10_full_zeroshot.json",
            dm / "eval_circle_gpt10_full.json",
        ),
    ]
    rows = [
        "three_gates\n(2–3 gate chains)",
        "five_gates\n(3–5 gate chains)",
        "mixed_tracks\n(chains ± circles)",
    ]
    M = np.array([[success(c[i + 1]) for c in cols] for i in range(3)])
    fig, ax = plt.subplots(figsize=(11.5, 3.4), constrained_layout=True)
    ax.grid(False)
    masked = np.ma.masked_invalid(M)
    ax.imshow(masked, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=GRID))
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if v > 0.55 else INK,
                )
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=7.5)
    ax.set_yticks(range(3))
    ax.set_yticklabels(rows, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Success rate of each checkpoint on each test distribution (1,024 fresh tracks, all gates required, deterministic + DR unless stated)"
        + (f"\n{TITLE_NOTE}" if TITLE_NOTE else ""),
        fontsize=9.5,
    )
    fig.savefig(out / f"{PREFIX}_fig2_eval_heatmap.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3: decoupling angle
def fig_decoupling(runs: dict[str, Path], out: Path):
    d5, dm = runs["five"], runs["mixed"]
    chains = [
        (0.0, d5 / "eval_5gates_det.json", 0.0),
        (3.7, d5 / "eval_chain_gyn013_zeroshot.json", 0.13),
        (7.5, d5 / "eval_chain_gyn026_zeroshot.json", 0.26),
        (15.0, d5 / "eval_chain_gyn052_zeroshot.json", 0.52),
    ]
    circles = [
        (19.4, d5 / "eval_circle_gpt36_zeroshot.json", 10),
        (14.6, d5 / "eval_circle_gpt24_zeroshot.json", 15),
        (15.3, d5 / "eval_circle_gpt12_zeroshot.json", 30),
        (18.0, d5 / "eval_circle_gpt10_full_zeroshot.json", 36),
        (22.5, d5 / "eval_circle_gpt8_zeroshot.json", 45),
        (30.0, d5 / "eval_circle_gpt6_zeroshot.json", 60),
        (36.1, d5 / "eval_circle_gpt5_zeroshot.json", 72),
    ]
    mixed = [
        (0.0, dm / "eval_chain5.json", "chain"),
        (8.6, dm / "eval_chain5_gyn03.json", "chain ±0.3 rad"),
        (15.3, dm / "eval_circle_gpt12.json", "30°"),
        (22.5, dm / "eval_circle_gpt8.json", "45°"),
        (30.0, dm / "eval_circle_gpt6.json", "60°"),
        (36.1, dm / "eval_circle_gpt5.json", "72°"),
        (45.1, dm / "eval_circle_gpt4.json", "90°"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    mk = dict(markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="none")
    xs, ys = zip(*[(a, success(p)) for a, p, _ in chains], strict=True)
    ax.plot(
        xs,
        ys,
        marker="o",
        color=SERIES[0],
        label="five_gates expert on chains, gate normals rotated ± noise",
        **mk,
    )
    for (a, p, n), y in zip(chains, ys, strict=True):
        if n:
            ax.annotate(
                f"±{n:.2f} rad",
                (a, y),
                xytext=(7, -3),
                textcoords="offset points",
                fontsize=7.5,
                color=INK2,
            )
    xs, ys = zip(*[(a, success(p)) for a, p, _ in circles], strict=True)
    ax.plot(
        xs, ys, marker="s", color=SERIES[1], label="five_gates expert on imperfect circles", **mk
    )
    for (a, p, deg), y in zip(circles, ys, strict=True):
        if deg in (10, 45, 72):
            dy = -14 if deg == 10 else 8
            ax.annotate(
                f"{deg}°/gate",
                (a, y),
                xytext=(0, dy),
                textcoords="offset points",
                fontsize=7.5,
                color=INK2,
                ha="center",
            )
    xs, ys = zip(*[(a, success(p)) for a, p, _ in mixed], strict=True)
    ax.plot(
        xs,
        ys,
        marker="^",
        color=SERIES[2],
        label="mixed_tracks expert (after the 1.4 min fine-tune)",
        **mk,
    )
    ax.annotate(
        "90°/gate, outside\nthe training range",
        (45.1, ys[-1]),
        xytext=(-8, -26),
        textcoords="offset points",
        fontsize=7.5,
        color=INK2,
        ha="right",
    )
    ax.set_xlabel(
        "mean angle between a gate's normal and the direction of travel into it (degrees, 4,096 sampled tracks)"
    )
    ax.set_ylabel("success rate (5 gates, deterministic, DR on)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-1.5, 48)
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.82))
    ax.set_title("Zero-shot success is a function of one geometric variable, not of curvature")
    fig.savefig(out / f"{PREFIX}_fig3_decoupling.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ figure 4: top-down trajectories
def rot2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def draw_track_panel(ax, d, worlds, title, cmap, vmin, vmax, half_opening):
    for k in worlds:
        n_active = int(d["n_active"][k])
        ep_len = int(d["ep_len"][k])
        gates = d["gates_pos"][k]
        yaws = d["gates_yaw"][k]
        pos = d["pos"][:ep_len, k]
        vel = d["vel"][:ep_len, k]
        # canonical frame: start at the origin heading +x
        # canonical heading = direction of the first segment (works for chains and circles alike)
        seg0 = gates[0, :2] - pos[0, :2]
        heading0 = np.arctan2(seg0[1], seg0[0])
        R = rot2d(-heading0)
        p = (R @ (pos[:, :2] - pos[0, :2]).T).T
        g = (R @ (gates[:, :2] - pos[0, :2]).T).T
        speed = np.linalg.norm(vel, axis=-1)
        segs = np.stack([p[:-1], p[1:]], axis=1)
        lc = LineCollection(
            segs,
            cmap=cmap,
            norm=plt.Normalize(vmin, vmax),
            linewidth=1.6,
            capstyle="round",
            zorder=3,
        )
        lc.set_array(speed[:-1])
        ax.add_collection(lc)
        for i in range(n_active + 1):
            yaw = float(yaws[i]) - heading0
            u = np.array([-np.sin(yaw), np.cos(yaw)])
            n = np.array([np.cos(yaw), np.sin(yaw)])
            a, b = g[i] - half_opening * u, g[i] + half_opening * u
            col = GATE_RED if i < n_active else DEEMPH
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=col,
                linewidth=2.4,
                solid_capstyle="butt",
                zorder=4,
                alpha=0.9,
            )
            ax.plot(
                [g[i, 0], g[i, 0] + 0.18 * n[0]],
                [g[i, 1], g[i, 1] + 0.18 * n[1]],
                color=col,
                linewidth=1.0,
                zorder=4,
                alpha=0.9,
            )
        if not d["success"][k]:
            ax.plot(
                p[-1, 0],
                p[-1, 1],
                marker="x",
                color=INK,
                markersize=6,
                markeredgewidth=1.4,
                zorder=5,
            )
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def fig_trajectories(runs: dict[str, Path], out: Path):
    d5 = np.load(runs["five"] / "traj_5gates.npz")
    dm_c = np.load(runs["mixed"] / "traj_circle8.npz")
    d5_c = np.load(runs["five"] / "traj_circle8.npz")
    half_opening = float(d5["half_opening"])
    vmin, vmax = 0.0, 6.0
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    draw_track_panel(
        axes[0],
        d5,
        range(6),
        "five_gates expert, 5-gate chains",
        SEQ_CMAP,
        vmin,
        vmax,
        half_opening,
    )
    draw_track_panel(
        axes[1],
        dm_c,
        range(4),
        "mixed_tracks expert, four 8-gate circles",
        SEQ_CMAP,
        vmin,
        vmax,
        half_opening,
    )
    draw_track_panel(
        axes[2],
        d5_c,
        range(4),
        "five_gates, same circles (× = failure)",
        SEQ_CMAP,
        vmin,
        vmax,
        half_opening,
    )
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.01)
    cb.set_label("speed (m/s)")
    cb.outline.set_visible(False)
    fig.suptitle(
        "Top-down trajectories, canonical frame (start at the origin, first gate along +x); red = required gates, grey = continuation gate; seed 0, so panels 2 and 3 share their tracks",
        fontsize=9.5,
        color=INK2,
        ha="left",
        x=0.01,
    )
    fig.savefig(out / f"{PREFIX}_fig4_trajectories.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5: one flight in detail
def fig_flight(runs: dict[str, Path], out: Path, world: int = 0):
    d = np.load(runs["mixed"] / "traj_circle8.npz")
    k, ep_len = world, int(d["ep_len"][world])
    fps = int(d["ctrl_freq"])
    t = np.arange(ep_len) / fps
    pos, vel, rpm, gp = (
        d["pos"][:ep_len, k],
        d["vel"][:ep_len, k],
        d["rpm_cmd"][:ep_len, k],
        d["gates_passed"][:ep_len, k],
    )
    hover = 18968.0
    speed = np.linalg.norm(vel, axis=-1)
    passes = np.where(np.diff(gp) > 0)[0] + 1
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 5.6), sharex=True, constrained_layout=True)
    axes[0].plot(t, pos[:, 2], color=SERIES[0])
    gz = d["gates_pos"][k][: int(d["n_active"][k]), 2]
    axes[0].plot(
        t[passes],
        gz[: len(passes)],
        marker="o",
        linestyle="none",
        color=SERIES[0],
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=1.2,
        zorder=4,
    )
    axes[0].set_ylabel("altitude (m)")
    axes[0].set_title("altitude (dots: gate altitude at each passage)")
    axes[1].plot(t, speed, color=SERIES[0])
    axes[1].set_ylabel("speed (m/s)")
    axes[1].set_title("speed")
    ratio = rpm / hover
    axes[2].fill_between(t, ratio.min(1), ratio.max(1), color=SERIES[0], alpha=0.12, linewidth=0)
    axes[2].plot(t, ratio.mean(1), color=SERIES[0])
    axes[2].axhline(1.0, color=AXIS, linewidth=0.8)
    axes[2].set_ylabel("commanded RPM / hover RPM")
    axes[2].set_title(
        "commanded RPM / hover — the VLA label (mean of the 4 motors; band = min–max)"
    )
    axes[2].set_xlabel("time (s)")
    for ax in axes:
        for i, p in enumerate(passes):
            ax.axvline(t[p], color=AXIS, linewidth=0.8, zorder=1)
        ax.set_xlim(0, t[-1])
    for i, p in enumerate(passes):
        axes[0].text(
            t[p],
            1.0,
            f" G{i + 1}",
            transform=axes[0].get_xaxis_transform(),
            fontsize=7.5,
            color=INK2,
            va="top",
        )
    fig.suptitle(
        f"One flight of the mixed_tracks expert on a full 8-gate circle (world {k} of the evaluation, the flight in the videos)",
        fontsize=9.5,
        color=INK2,
        ha="left",
        x=0.01,
    )
    fig.savefig(out / f"{PREFIX}_fig5_flight.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ figure 6: the long-chain lesson
def fig_long_chains(runs: dict[str, Path], out: Path):
    unc = np.load(runs["five"] / "traj_10gates.npz")
    cap = np.load(runs["five"] / "traj_10gates_capped.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)

    def failure_hist(d):
        gp, ep_len, succ = d["gates_passed"], d["ep_len"], d["success"]
        last = np.array([gp[ep_len[i] - 1, i] for i in range(len(succ))]).astype(int)
        return np.bincount(last[succ == 0], minlength=11)[:10] / len(succ)

    def speed_by_gate(d):
        gp, ep_len, vel = d["gates_passed"], d["ep_len"], d["vel"]
        T, K = gp.shape
        valid = np.arange(T)[:, None] < ep_len[None, :]
        speed = np.linalg.norm(vel, axis=-1)
        return np.array(
            [
                speed[valid & (gp == g)].mean() if (valid & (gp == g)).any() else np.nan
                for g in range(10)
            ]
        )

    hu, hc = failure_hist(unc), failure_hist(cap)
    xs = np.arange(10)
    w = 0.36
    for x, hu_i, hc_i in zip(xs, hu, hc, strict=True):
        rounded_bar(axes[0], x - w / 2 - 0.02, hu_i, w, SERIES[0])
        rounded_bar(axes[0], x + w / 2 + 0.02, hc_i, w, SERIES[1])
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([str(i) for i in xs])
    axes[0].set_xlabel("gates passed when the episode failed (10-gate chains)")
    axes[0].set_ylabel("fraction of all episodes")
    axes[0].set_ylim(0, max(hu.max(), hc.max()) * 1.25)
    axes[0].set_xlim(-0.6, 9.6)
    axes[0].legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=SERIES[0],
                linewidth=6,
                label=f"raw 'gates remaining' feature: success {unc['success'].mean():.2f}",
            ),
            plt.Line2D(
                [],
                [],
                color=SERIES[1],
                linewidth=6,
                label=f"feature capped at 5: success {cap['success'].mean():.2f}",
            ),
        ],
        loc="upper right",
    )
    axes[0].set_title("Where 10-gate episodes fail (same checkpoint, 256 tracks each)")
    su, sc = speed_by_gate(unc), speed_by_gate(cap)
    mk = dict(marker="o", markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2)
    axes[1].plot(xs + 1, su, color=SERIES[0], label="raw feature", **mk)
    axes[1].plot(xs + 1, sc, color=SERIES[1], label="capped feature", **mk)
    axes[1].set_xlabel("gate being approached")
    axes[1].set_ylabel("mean speed (m/s)")
    axes[1].set_ylim(0, 7)
    axes[1].set_xticks(xs + 1)
    axes[1].legend(loc="lower right")
    axes[1].set_title("Speed plateaus by gate 3, then falls")
    fig.savefig(out / f"{PREFIX}_fig6_long_chains.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ figure 7: rendered frames
def fig_frames(frames: dict[str, Path], out: Path):
    from PIL import Image

    order = [
        ("overview", "Mixed-track expert, 8-gate circle: overview (yellow = path, cyan = drone)"),
        ("track", "Same flight: chase camera"),
        ("fpv", "Same flight: onboard FPV camera (the Phase-2 observation)"),
        ("fail", "Chain-only expert, same kind of circle: along the normal, into frame 2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), order, strict=True):
        ax.imshow(np.asarray(Image.open(frames[key]).convert("RGB")))
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    fig.savefig(out / f"{PREFIX}_fig7_frames.png", dpi=150)
    plt.close(fig)


def latest(pattern: str) -> Path:
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not paths:
        raise SystemExit(f"no run matches {pattern}")
    return Path(paths[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--three", type=Path, default=None)
    p.add_argument("--five", type=Path, default=None)
    p.add_argument("--mixed", type=Path, default=None)
    p.add_argument(
        "--frames",
        type=Path,
        default=None,
        help="dir with overview.png, track.png, fpv.png, fail.png",
    )
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    p.add_argument("--prefix", default="phase1", help="figure file prefix")
    p.add_argument("--title-note", default="", help="second line of the heatmap title")
    p.add_argument(
        "--only",
        nargs="*",
        choices=["training", "heatmap", "decoupling", "trajectories", "flight", "long_chains"],
        default=None,
        help="build only these figures (the last three need trajectory .npz files)",
    )
    args = p.parse_args()
    global PREFIX, TITLE_NOTE
    PREFIX, TITLE_NOTE = args.prefix, args.title_note
    runs = {
        "three": args.three or latest("outputs/three_gates/*/"),
        "five": args.five or latest("outputs/five_gates/*/"),
        "mixed": args.mixed or latest("outputs/mixed_tracks/*/"),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    figures = {
        "training": fig_training,
        "heatmap": fig_heatmap,
        "decoupling": fig_decoupling,
        "trajectories": fig_trajectories,
        "flight": fig_flight,
        "long_chains": fig_long_chains,
    }
    for name, fn in figures.items():
        if args.only is None or name in args.only:
            fn(runs, args.out)
    if args.frames:
        fig_frames(
            {k: args.frames / f"{k}.png" for k in ("overview", "track", "fpv", "fail")}, args.out
        )
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
