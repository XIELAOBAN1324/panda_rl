"""Pure geometry tests for the fixed directed corner convention."""

import numpy as np

from panda_mujoco_gym.envs.window_assembly_geometry import (
    alignment_errors_from_corners,
    build_frame_from_corners,
    fine_alignment_costs,
    ordered_rectangle_corners,
    so3_exp,
    so3_log,
)


def _corners(center=None, rotation=None):
    center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    return ordered_rectangle_corners(center, rotation, 0.18, 0.27)


def _errors(center=None, rotation=None):
    return alignment_errors_from_corners(_corners(center, rotation), _corners())


def test_corner_order_and_built_frame_are_orthonormal():
    rotation = so3_exp(np.array([0.3, -0.2, 0.4]))
    corners = _corners([0.2, -0.1, 0.8], rotation)
    center, rebuilt = build_frame_from_corners(corners)
    assert np.allclose(center, [0.2, -0.1, 0.8])
    assert np.allclose(rebuilt.T @ rebuilt, np.eye(3), atol=1e-12)
    assert np.linalg.det(rebuilt) > 0.999999
    assert np.dot(corners[1] - corners[0], rebuilt[:, 1]) > 0.0
    assert np.dot(corners[0] - corners[3], rebuilt[:, 0]) > 0.0


def test_left_right_and_up_down_offsets_have_opposite_signs():
    left = _errors(center=[-0.01, 0.0, 0.0])
    right = _errors(center=[0.01, 0.0, 0.0])
    down = _errors(center=[0.0, -0.01, 0.0])
    up = _errors(center=[0.0, 0.01, 0.0])
    assert left["error_u"] < 0.0 < right["error_u"]
    assert down["error_v"] < 0.0 < up["error_v"]


def test_positive_and_negative_t1_tilts_have_opposite_signs():
    positive = _errors(rotation=so3_exp(np.array([0.02, 0.0, 0.0])))
    negative = _errors(rotation=so3_exp(np.array([-0.02, 0.0, 0.0])))
    assert positive["tilt_error_t1"] > 0.0 > negative["tilt_error_t1"]
    assert abs(float(positive["tilt_error_t2"])) < 1e-12


def test_positive_and_negative_t2_tilts_have_opposite_signs():
    positive = _errors(rotation=so3_exp(np.array([0.0, 0.02, 0.0])))
    negative = _errors(rotation=so3_exp(np.array([0.0, -0.02, 0.0])))
    assert positive["tilt_error_t2"] > 0.0 > negative["tilt_error_t2"]
    assert abs(float(positive["tilt_error_t1"])) < 1e-12


def test_positive_and_negative_yaw_have_opposite_signs():
    positive = _errors(rotation=so3_exp(np.array([0.0, 0.0, 0.03])))
    negative = _errors(rotation=so3_exp(np.array([0.0, 0.0, -0.03])))
    assert positive["yaw_error"] > 0.0 > negative["yaw_error"]


def test_translation_does_not_create_rotation_error():
    errors = _errors(center=[0.012, -0.009, 0.08])
    assert np.allclose(
        [errors["tilt_error_t1"], errors["tilt_error_t2"], errors["yaw_error"]],
        0.0,
        atol=1e-12,
    )


def test_rotation_about_center_does_not_create_translation_error():
    errors = _errors(rotation=so3_exp(np.array([0.01, -0.02, 0.03])))
    assert abs(float(errors["error_u"])) < 1e-12
    assert abs(float(errors["error_v"])) < 1e-12


def test_perfect_alignment_has_zero_five_dof_error():
    errors = _errors()
    assert np.allclose(
        [
            errors["error_u"],
            errors["error_v"],
            errors["tilt_error_t1"],
            errors["tilt_error_t2"],
            errors["yaw_error"],
        ],
        0.0,
        atol=1e-12,
    )


def test_so3_log_is_finite_near_zero_and_pi_and_orthogonalizes_input():
    assert np.allclose(so3_log(so3_exp([1e-10, -2e-10, 3e-10])), [1e-10, -2e-10, 3e-10], atol=1e-9)
    near_pi = so3_log(so3_exp([np.pi - 1e-7, 0.0, 0.0]))
    assert np.all(np.isfinite(near_pi))
    noisy = so3_exp([0.01, -0.02, 0.03])
    noisy[0, 0] += 1e-7
    assert np.all(np.isfinite(so3_log(noisy)))


def test_worst_cost_exposes_one_bad_degree_of_freedom():
    errors = {
        "error_u": 0.0,
        "error_v": 0.0,
        "tilt_error_t1": 0.0,
        "tilt_error_t2": 0.0,
        "yaw_error": np.deg2rad(5.0),
        "normal_gap_error": 0.0,
    }
    costs = fine_alignment_costs(
        errors,
        translation_tolerance=0.002,
        tilt_tolerance_rad=np.deg2rad(0.5),
        yaw_tolerance_rad=np.deg2rad(0.5),
        normal_gap_tolerance=0.001,
    )
    assert costs["worst_alignment_cost"] > costs["mean_alignment_cost"]
    assert costs["state_cost"] > 0.0
