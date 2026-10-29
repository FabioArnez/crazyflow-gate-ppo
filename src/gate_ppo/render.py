"""Replay a trajectory ``.npz`` (from ``gate-ppo-eval --save-traj``) into an MP4 with gates drawn.

Rendering is crazyflow's per-world MuJoCo offscreen path, i.e. a second pass over logged poses —
the same pattern Phase 2 will use to produce FPV frames for the demonstration dataset.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from crazyflow.control import Control  # noqa: E402
from crazyflow.dynamics import Dynamics  # noqa: E402
from crazyflow.sim import Sim  # noqa: E402

CAMERAS = {"track": "track_cam:0", "fpv": "fpv_cam:0", "overview": "overview", "free": -1}


def add_gates(sim: Sim, gates_pos, gates_yaw, half_opening: float, thickness: float, n_active: int):
    """Attach square gate frames (4 boxes each) to the scene and rebuild the MJX model."""
    w, th, depth = float(half_opening), float(thickness), 0.03
    bars = [
        ((0.0, 0.0, w + th / 2), (depth / 2, w + th, th / 2)),
        ((0.0, 0.0, -(w + th / 2)), (depth / 2, w + th, th / 2)),
        ((0.0, w + th / 2, 0.0), (depth / 2, th / 2, w)),
        ((0.0, -(w + th / 2), 0.0), (depth / 2, th / 2, w)),
    ]
    for i, (pos, yaw) in enumerate(zip(gates_pos, gates_yaw, strict=True)):
        body = sim.spec.worldbody.add_body()
        body.name = f"gate{i}"
        body.pos = np.asarray(pos, dtype=float)
        body.quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])  # MuJoCo wxyz
        rgba = [0.9, 0.2, 0.2, 1.0] if i < n_active else [0.6, 0.6, 0.6, 0.6]
        for bar_pos, size in bars:
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.pos = np.asarray(bar_pos)
            geom.size = np.asarray(size)
            geom.rgba = np.asarray(rgba)
            geom.contype = 0
            geom.conaffinity = 0
    sim.build_mjx()


def add_overview_camera(sim: Sim, gates_pos, name: str = "overview") -> None:
    """Add a fixed camera that frames the whole track from above, then rebuild the MJX model."""
    from scipy.spatial.transform import Rotation  # after crazyflow, which sets SCIPY_ARRAY_API

    pts = np.asarray(gates_pos, dtype=float)
    centre = pts.mean(0)
    xy = pts[:, :2] - centre[:2]
    extent = max(float(np.linalg.norm(xy, axis=1).max()), 1.5)
    azimuth = np.deg2rad(-135.0)
    dist, height = 1.9 * extent + 1.2, 1.3 * extent + 1.0
    # Elongated tracks (long chains) are viewed obliquely from the side of their main axis, from
    # behind the start, so that they are not foreshortened and the gate openings stay visible;
    # round tracks keep the default diagonal view.
    _, sv, vt = np.linalg.svd(xy, full_matrices=False)
    if sv[0] > 2.0 * sv[1]:
        axis = vt[0] if (pts[-1, :2] - pts[0, :2]) @ vt[0] > 0 else -vt[0]
        azimuth = np.arctan2(axis[1], axis[0]) - np.deg2rad(125.0)
        dist, height = 1.2 * extent + 1.5, 0.8 * extent + 1.0
    pos = centre + np.array([dist * np.cos(azimuth), dist * np.sin(azimuth), height])
    fwd = centre - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # MuJoCo cameras look along -z with +y up: columns are the camera x, y, z axes in the world.
    rot = np.stack([right, up, -fwd], axis=1)
    cam = sim.spec.worldbody.add_camera()
    cam.name = name
    cam.pos = pos
    cam.quat = Rotation.from_matrix(rot).as_quat(scalar_first=True)
    cam.fovy = 50.0
    sim.build_mjx()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--traj", required=True)
    p.add_argument("--out", default="flight.mp4")
    p.add_argument("--world", type=int, default=0)
    p.add_argument("--camera", choices=sorted(CAMERAS), default="track")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--trail",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="draw the flown path and a halo on the drone (default: only for the overview camera)",
    )
    args = p.parse_args(argv)

    d = np.load(args.traj)
    k = args.world
    ep_len = int(d["ep_len"][k])
    pos, quat = d["pos"][:ep_len, k], d["quat"][:ep_len, k]
    n_active = int(d["n_active"][k])
    sim = Sim(
        n_worlds=1,
        n_drones=1,
        drone=str(d["drone"]),
        dynamics=Dynamics.first_principles,
        control=Control.rotor_vel,
        freq=int(d["sim_freq"]),
        device="cpu",
    )
    add_gates(
        sim, d["gates_pos"][k], d["gates_yaw"][k], d["half_opening"], d["frame_thickness"], n_active
    )
    if args.camera == "overview":
        add_overview_camera(sim, d["gates_pos"][k])
    sim.reset()

    ctrl_freq = int(d["ctrl_freq"])
    stride = max(1, round(ctrl_freq / args.fps))
    trail = args.camera == "overview" if args.trail is None else args.trail
    render = lambda: sim.render(
        mode="rgb_array", world=0, camera=CAMERAS[args.camera], width=args.width, height=args.height
    )
    render()  # instantiate the offscreen viewer so markers can be attached to it
    viewer = sim.viewer.viewer
    frames = []
    for t in range(0, ep_len, stride):
        states = sim.data.states.replace(
            pos=jnp.asarray(pos[t]).reshape(1, 1, 3), quat=jnp.asarray(quat[t]).reshape(1, 1, 4)
        )
        core = sim.data.core.replace(mjx_synced=jnp.array(False))
        sim.data = sim.data.replace(states=states, core=core)
        if trail:
            markers = getattr(viewer, "_markers", None)
            if markers is not None:
                markers[:] = []
            for s in range(0, t, max(1, stride // 2)):
                viewer.add_marker(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.full(3, 0.025),
                    pos=pos[s],
                    rgba=np.array([1.0, 0.85, 0.1, 0.9]),
                )
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.full(3, 0.07),
                pos=pos[t],
                rgba=np.array([0.2, 0.9, 1.0, 0.8]),
            )
        frames.append(np.asarray(render()))
    sim.close()
    imageio.mimwrite(args.out, frames, fps=ctrl_freq // stride)
    outcome = "success" if d["success"][k] else "failure"
    print(
        f"{len(frames)} frames ({ep_len / ctrl_freq:.1f} s, {outcome}, {n_active} gates) "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
