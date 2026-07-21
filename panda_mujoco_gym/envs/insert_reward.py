import numpy as np


def _safe_clip01(value):
    return np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)


def compute_insert_dense_reward(
    orientation_alignment,
    inplane_alignment,
    glass_fits_window,
    fit_margin,
    insert_depth,
    inplane_offset,
    prealign_distance,
    preinsert_distance,
    final_insert_distance,
    reward_stage,
    collision,
    plane_violation,
    orientation_threshold: float,
):
    """Dense shaping aligned with the true window-insertion success condition."""

    orientation_alignment = np.asarray(orientation_alignment, dtype=np.float32)
    inplane_alignment = np.asarray(inplane_alignment, dtype=np.float32)
    glass_fits_window = np.asarray(glass_fits_window, dtype=np.float32)
    fit_margin = np.asarray(fit_margin, dtype=np.float32)
    insert_depth = np.asarray(insert_depth, dtype=np.float32)
    inplane_offset = np.asarray(inplane_offset, dtype=np.float32)
    prealign_distance = np.asarray(prealign_distance, dtype=np.float32)
    preinsert_distance = np.asarray(preinsert_distance, dtype=np.float32)
    final_insert_distance = np.asarray(final_insert_distance, dtype=np.float32)
    reward_stage = np.asarray(reward_stage, dtype=np.float32)
    collision = np.asarray(collision, dtype=np.float32)
    plane_violation = np.asarray(plane_violation, dtype=np.float32)

    orientation_denominator = max(1.0 - float(orientation_threshold), 1e-6)
    stage_insert = reward_stage > 0.5

    true_success = (
        stage_insert
        & (glass_fits_window > 0.0)
        & (collision <= 0.0)
        & (plane_violation <= 0.0)
        & (final_insert_distance < 0.006)
    )

    align_quality = np.minimum(
        _safe_clip01((orientation_alignment - orientation_threshold) / orientation_denominator),
        _safe_clip01((inplane_alignment - orientation_threshold) / orientation_denominator),
    )
    fit_quality = _safe_clip01((fit_margin + 0.03) / 0.06)
    offset_quality = 1.0 - _safe_clip01(inplane_offset / 0.06)
    depth_quality = _safe_clip01(insert_depth)
    prealign_quality = 1.0 - _safe_clip01(prealign_distance / 0.22)
    preinsert_quality = 1.0 - _safe_clip01(preinsert_distance / 0.10)
    final_insert_quality = 1.0 - _safe_clip01(final_insert_distance / 0.02)
    stage_bonus = stage_insert.astype(np.float32)
    near_window_gate = np.maximum(prealign_quality, preinsert_quality)
    fit_focus_gate = np.maximum(preinsert_quality, stage_bonus)
    fit_insert_gate = _safe_clip01((fit_margin + 0.02) / 0.03)
    insert_gate = (
        align_quality
        * offset_quality
        * fit_insert_gate
    ).astype(np.float32)
    improper_push = (1.0 - insert_gate) * depth_quality
    unfit_penalty = fit_focus_gate * _safe_clip01((-fit_margin) / 0.03)

    align_reward = (
        0.18 * prealign_quality
        + 0.18 * preinsert_quality
        + 0.14 * align_quality
        + 0.12 * offset_quality
        + 0.08 * near_window_gate * fit_quality
    )
    insert_reward = (
        0.30 * insert_gate * depth_quality
        + 0.16 * fit_quality
        + 0.16 * final_insert_quality
        + 0.08 * align_quality
        + 0.04 * offset_quality
        + 0.04 * preinsert_quality
        + 0.04 * stage_bonus
    )

    reward = np.where(stage_insert, insert_reward, align_reward + 0.06 * insert_gate).astype(np.float32)
    reward -= 0.30 * improper_push
    reward -= 0.14 * _safe_clip01(inplane_offset / 0.08)
    reward -= 0.18 * unfit_penalty
    reward += 0.30 * true_success.astype(np.float32)
    reward -= 1.00 * collision
    reward -= 1.00 * plane_violation
    return reward.astype(np.float32)
