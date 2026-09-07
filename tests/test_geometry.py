import jax.numpy as jnp
import numpy as np

from gate_ppo.geometry import (
    gate_crossing,
    gate_frame,
    hover_rpm,
    quat_from_rpy,
    quat_rotate,
    quat_rotate_inv,
)


def test_quat_from_yaw_rotates_x_axis():
    yaw = jnp.array(0.7)
    q = quat_from_rpy(jnp.array(0.0), jnp.array(0.0), yaw)
    x_world = quat_rotate(q, jnp.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(x_world, [np.cos(0.7), np.sin(0.7), 0.0], atol=1e-6)


def test_quat_rotate_inverse_roundtrip():
    q = quat_from_rpy(jnp.array(0.3), jnp.array(-0.2), jnp.array(1.1))
    v = jnp.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(quat_rotate_inv(q, quat_rotate(q, v)), v, atol=1e-6)


def test_gate_frame_is_orthonormal():
    n, u = gate_frame(jnp.array(0.4))
    assert abs(float(jnp.dot(n, u))) < 1e-6
    np.testing.assert_allclose(jnp.linalg.norm(n), 1.0, atol=1e-6)


def test_gate_crossing_classification():
    gate = jnp.zeros(3)
    yaw = jnp.array(0.0)  # normal along +x
    kw = dict(half_opening=0.245, frame_thickness=0.06, drone_radius=0.07)

    def classify(prev, pos):
        out = gate_crossing(jnp.array(prev), jnp.array(pos), gate, yaw, **kw)
        return tuple(bool(x) for x in out)

    assert classify([-0.1, 0.0, 0.0], [0.1, 0.05, 0.0]) == (
        True,
        False,
        False,
    )  # through the opening
    assert classify([-0.1, 0.3, 0.0], [0.1, 0.3, 0.0]) == (
        False,
        True,
        False,
    )  # into the frame band
    assert classify([-0.1, 0.0, 0.3], [0.1, 0.0, 0.3]) == (False, True, False)  # top bar
    assert classify([-0.1, 0.6, 0.0], [0.1, 0.6, 0.0]) == (False, False, True)  # beside the gate
    assert classify([-0.3, 0.0, 0.0], [-0.1, 0.0, 0.0]) == (False, False, False)  # no crossing
    assert classify([0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]) == (False, False, False)  # backwards


def test_gate_crossing_uses_interpolated_point():
    # Lateral offset at the crossing point (x=0) is 0.1 even though the endpoints are far off-axis.
    gate = jnp.zeros(3)
    passed, hit, missed = gate_crossing(
        jnp.array([-1.0, -0.4, 0.0]),
        jnp.array([1.0, 0.6, 0.0]),
        gate,
        jnp.array(0.0),
        half_opening=0.245,
        frame_thickness=0.06,
        drone_radius=0.07,
    )
    assert bool(passed) and not bool(hit) and not bool(missed)


def test_hover_rpm_cf2x_l250_matches_vault():
    rpm = hover_rpm([0.0, -5.382196214637237e-7, 2.4582929831265485e-10], mass=0.0319)
    assert abs(rpm - 18968.0) < 5.0
