"""Geometry and reward helpers for window fine alignment.

Corner order is always ``top_left, top_right, bottom_right, bottom_left``.
Rotation matrices use columns ``[t1, t2, n]`` and are projected onto SO(3)
before relative rotations are evaluated.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")


def normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a normalized float64 vector, rejecting degenerate inputs."""

    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError("Cannot normalize a degenerate vector")
    return value / norm


def project_to_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Project a nearly-orthogonal 3x3 matrix onto SO(3) with an SVD."""

    value = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(value)):
        raise ValueError("Rotation matrix contains NaN or Inf")
    u, _, vh = np.linalg.svd(value)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def ordered_rectangle_corners(
    center_world: np.ndarray,
    rotation_world: np.ndarray,
    half_extent_t1: float,
    half_extent_t2: float,
) -> np.ndarray:
    """Create ordered rectangle corners in world coordinates."""

    center = np.asarray(center_world, dtype=np.float64).reshape(3)
    rotation = project_to_rotation_matrix(rotation_world)
    t1 = rotation[:, 0]
    t2 = rotation[:, 1]
    half_t1 = float(half_extent_t1)
    half_t2 = float(half_extent_t2)
    if half_t1 <= 0.0 or half_t2 <= 0.0:
        raise ValueError("Rectangle half extents must be positive")
    return np.stack(
        [
            center + half_t1 * t1 - half_t2 * t2,
            center + half_t1 * t1 + half_t2 * t2,
            center - half_t1 * t1 + half_t2 * t2,
            center - half_t1 * t1 - half_t2 * t2,
        ],
        axis=0,
    )


def build_frame_from_corners(corners_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return rectangle center and an SO(3) frame with columns ``[t1,t2,n]``.

    ``t1`` points from the bottom edge to the top edge, while ``t2`` points
    from the left edge to the right edge. The fixed directed corner order is
    therefore part of the state definition and is never rematched by distance.
    """

    corners = np.asarray(corners_world, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError(f"Expected corners with shape (4, 3), got {corners.shape}")
    if not np.all(np.isfinite(corners)):
        raise ValueError("Corners contain NaN or Inf")

    top_center = 0.5 * (corners[0] + corners[1])
    bottom_center = 0.5 * (corners[2] + corners[3])
    left_center = 0.5 * (corners[0] + corners[3])
    right_center = 0.5 * (corners[1] + corners[2])

    t1 = normalize_vector(top_center - bottom_center)
    raw_t2 = right_center - left_center
    raw_t2 = raw_t2 - float(np.dot(raw_t2, t1)) * t1
    t2 = normalize_vector(raw_t2)
    normal = normalize_vector(np.cross(t1, t2))
    t2 = normalize_vector(np.cross(normal, t1))
    rotation = project_to_rotation_matrix(np.column_stack([t1, t2, normal]))
    center = np.mean(corners, axis=0)
    return center, rotation


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Stable SO(3) exponential map for a three-dimensional rotation vector."""

    vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Rotation vector contains NaN or Inf")
    angle = float(np.linalg.norm(vector))
    skew = np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return project_to_rotation_matrix(np.eye(3) + skew + 0.5 * (skew @ skew))
    angle_sq = angle * angle
    return project_to_rotation_matrix(
        np.eye(3)
        + (np.sin(angle) / angle) * skew
        + ((1.0 - np.cos(angle)) / angle_sq) * (skew @ skew)
    )


def so3_log(rotation_matrix: np.ndarray) -> np.ndarray:
    """Stable SO(3) logarithm, including small angles and angles near pi."""

    rotation = project_to_rotation_matrix(rotation_matrix)
    cos_angle = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * vee
    if np.pi - angle < 1e-5:
        symmetric = 0.5 * (rotation + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
        dominant = int(np.argmax(axis))
        if axis[dominant] < 1e-8:
            eigenvalues, eigenvectors = np.linalg.eigh(rotation)
            axis = eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))]
        else:
            for index in range(3):
                if index != dominant:
                    axis[index] = symmetric[dominant, index] / axis[dominant]
        axis = normalize_vector(axis)
        if float(np.dot(axis, vee)) < 0.0:
            axis = -axis
        return axis * angle
    return (angle / (2.0 * np.sin(angle))) * vee


def alignment_errors_from_corners(
    glass_corners_world: np.ndarray,
    frame_corners_world: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compute the five controllable alignment errors from ordered corners."""

    glass_center, glass_rotation = build_frame_from_corners(glass_corners_world)
    frame_center, frame_rotation = build_frame_from_corners(frame_corners_world)
    center_delta = glass_center - frame_center
    relative_rotation = frame_rotation.T @ glass_rotation
    rotation_error = so3_log(relative_rotation)
    corner_distances = np.linalg.norm(
        np.asarray(glass_corners_world, dtype=np.float64)
        - np.asarray(frame_corners_world, dtype=np.float64),
        axis=1,
    )
    return {
        "error_u": float(np.dot(frame_rotation[:, 0], center_delta)),
        "error_v": float(np.dot(frame_rotation[:, 1], center_delta)),
        "tilt_error_t1": float(rotation_error[0]),
        "tilt_error_t2": float(rotation_error[1]),
        "yaw_error": float(rotation_error[2]),
        "corner_distances": corner_distances,
        "glass_center_world": glass_center,
        "frame_center_world": frame_center,
        "glass_rotation_world": glass_rotation,
        "frame_rotation_world": frame_rotation,
        "glass_normal_world": glass_rotation[:, 2],
        "frame_normal_world": frame_rotation[:, 2],
    }


def huber_scalar(value: float, delta: float = 1.0) -> float:
    """Scalar Huber loss."""

    absolute = abs(float(value))
    threshold = float(delta)
    if absolute <= threshold:
        return 0.5 * absolute * absolute
    return threshold * (absolute - 0.5 * threshold)


def stable_smooth_max_abs(values: np.ndarray, kappa: float = 5.0) -> float:
    """Return ``log(mean(exp(kappa*abs(values)))) / kappa`` stably."""

    scaled = float(kappa) * np.abs(np.asarray(values, dtype=np.float64))
    maximum = float(np.max(scaled))
    return float((maximum + np.log(np.mean(np.exp(scaled - maximum)))) / float(kappa))


def fine_alignment_costs(
    errors: Mapping[str, float | np.ndarray],
    *,
    translation_tolerance: float,
    tilt_tolerance_rad: float,
    yaw_tolerance_rad: float,
    normal_gap_tolerance: float,
    smooth_max_kappa: float = 5.0,
) -> dict[str, float | np.ndarray]:
    """Compute alignment and fixed preinsert-plane gap costs."""

    normalized = np.array(
        [
            float(errors["error_u"]) / translation_tolerance,
            float(errors["error_v"]) / translation_tolerance,
            float(errors["tilt_error_t1"]) / tilt_tolerance_rad,
            float(errors["tilt_error_t2"]) / tilt_tolerance_rad,
            float(errors["yaw_error"]) / yaw_tolerance_rad,
        ],
        dtype=np.float64,
    )
    mean_cost = float(np.mean([huber_scalar(value) for value in normalized]))
    worst_cost = stable_smooth_max_abs(normalized, kappa=smooth_max_kappa)
    alignment_cost = 0.7 * mean_cost + 0.3 * worst_cost
    normal_gap_error = float(
        errors["normal_gap_error"] if "normal_gap_error" in errors else errors["normal_gap_drift"]
    )
    normalized_gap = normal_gap_error / normal_gap_tolerance
    gap_cost = huber_scalar(normalized_gap)
    state_cost = alignment_cost + 0.1 * gap_cost
    return {
        "normalized_alignment_error": normalized,
        "mean_alignment_cost": mean_cost,
        "worst_alignment_cost": worst_cost,
        "alignment_cost": alignment_cost,
        "gap_cost": gap_cost,
        "state_cost": state_cost,
    }
