import jax
import jax.numpy as jnp
import numpy as np

from gate_ppo.track import sample_circle_track


def test_circle_track_turns_and_chords():
    gates_per_turn = 8.0
    gates_pos, yaws, start_pos, start_yaw = sample_circle_track(
        jax.random.key(3),
        n_gates=8,
        gates_per_turn_min=jnp.float32(gates_per_turn),
        gates_per_turn_max=jnp.float32(gates_per_turn),
        spacing_min=1.5,
        spacing_max=1.5,
        radius_noise=0.0,
        angle_noise=0.0,
        yaw_noise=0.0,
        dz_max=0.0,
        z_min=0.7,
        z_max=2.2,
        start_z_min=1.0,
        start_z_max=1.0,
    )
    gates_pos, yaws = np.asarray(gates_pos), np.asarray(yaws)
    pts = np.concatenate([np.asarray(start_pos)[None, :2], gates_pos[:, :2]])
    chords = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    np.testing.assert_allclose(chords, 1.5, atol=1e-4)  # chord == spacing
    turns = np.diff(np.concatenate([[float(start_yaw)], yaws]))
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(np.abs(turns), 2 * np.pi / gates_per_turn, atol=1e-4)
    assert np.all(np.sign(turns) == np.sign(turns[0]))  # constant turn direction
    # 8 gates at 45 degrees each close the circle: the last gate sits on the spawn point.
    np.testing.assert_allclose(gates_pos[-1, :2], 0.0, atol=1e-3)
