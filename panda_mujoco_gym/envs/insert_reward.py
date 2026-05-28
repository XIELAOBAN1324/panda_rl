import numpy as np


def compute_insert_dense_reward(
    distances,
    orientation_error,
    inplane_error,
    orientation_alignment,
    inplane_alignment,
    glass_fits_window,
    collision,
    plane_violation,
    distance_threshold: float,
    orientation_threshold: float,
):
    """Dense shaping aligned with the true window-insertion success condition."""

    distances = np.asarray(distances, dtype=np.float32)
    orientation_error = np.asarray(orientation_error, dtype=np.float32)
    inplane_error = np.asarray(inplane_error, dtype=np.float32)
    orientation_alignment = np.asarray(orientation_alignment, dtype=np.float32)
    inplane_alignment = np.asarray(inplane_alignment, dtype=np.float32)
    glass_fits_window = np.asarray(glass_fits_window, dtype=np.float32)
    collision = np.asarray(collision, dtype=np.float32)
    plane_violation = np.asarray(plane_violation, dtype=np.float32)

    distance_threshold = max(float(distance_threshold), 1e-6)
    orientation_denominator = max(1.0 - float(orientation_threshold), 1e-6)

    near_pose = (
        (distances < distance_threshold)
        & (orientation_alignment >= orientation_threshold)
        & (inplane_alignment >= orientation_threshold)
    )
    true_success = (
        near_pose
        & (glass_fits_window > 0.0)
        & (collision <= 0.0)
        & (plane_violation <= 0.0)
    )

    position_cost = np.minimum(distances / distance_threshold, 5.0)
    orientation_cost = np.minimum(orientation_error / orientation_denominator, 5.0)
    inplane_cost = np.minimum(inplane_error / orientation_denominator, 5.0)

    reward = -0.08 * position_cost
    reward -= 0.04 * orientation_cost
    reward -= 0.04 * inplane_cost
    reward += 0.25 * true_success.astype(np.float32)
    reward -= 0.20 * (near_pose & ~(glass_fits_window > 0.0)).astype(np.float32)
    reward -= 1.00 * collision
    reward -= 1.00 * plane_violation
    return reward.astype(np.float32)
