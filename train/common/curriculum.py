"""Pick-and-place sparse 학습용 aggressive curriculum wrapper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    progress_upper: float
    object_xy_range: float
    goal_radius_min: float
    goal_radius_max: float
    goal_min_z: float
    goal_max_z: float
    air_goal_prob: float
    distance_threshold: float
    shaping_scale: float
    spawn_ee_above_object: bool
    ee_hover_height: float
    use_env_default_goal: bool = False
    settle_steps: int = 1


class PickAndPlaceCurriculumWrapper(gym.Wrapper):
    """FrankaPickAndPlaceSparse를 더 쉽게 시작시키는 aggressive curriculum.

    핵심 아이디어
    - 초기에는 object spawn 범위를 좁힌다.
    - 초기에는 goal 을 object 근처/위쪽에 두어 '잡고 살짝 들기'를 먼저 학습시킨다.
    - episode 시작 시 EE를 object 위로 옮겨 초기 탐색 난이도를 크게 낮춘다.
    - sparse reward 위에 작은 shaping bonus 를 얹되, 진행될수록 0으로 감쇠시킨다.
    - 마지막 stage 에서는 원래 full task 분포와 원래 sparse reward로 돌아간다.
    """

    def __init__(self, env: gym.Env, total_env_steps_target: int = 1_000_000):
        super().__init__(env)
        self.total_env_steps_target = max(int(total_env_steps_target), 1)
        self.local_env_steps = 0
        self.episode_idx = 0
        self.current_stage: CurriculumStage | None = None

        base = self.unwrapped
        self.base_obj_xy_range = float(getattr(base, "obj_xy_range", 0.3))
        self.base_goal_xy_range = float(getattr(base, "goal_xy_range", 0.3))
        self.base_goal_z_range = float(getattr(base, "goal_z_range", 0.2))
        self.base_distance_threshold = float(getattr(base, "distance_threshold", 0.05))
        self.base_obj_range_low = np.array(getattr(base, "obj_range_low"), dtype=np.float64).copy()
        self.base_obj_range_high = np.array(getattr(base, "obj_range_high"), dtype=np.float64).copy()
        self.base_goal_range_low = np.array(getattr(base, "goal_range_low"), dtype=np.float64).copy()
        self.base_goal_range_high = np.array(getattr(base, "goal_range_high"), dtype=np.float64).copy()
        self.base_obj_center = (self.base_obj_range_low + self.base_obj_range_high) / 2.0
        self.base_goal_center = (self.base_goal_range_low + self.base_goal_range_high) / 2.0

        self.stages: List[CurriculumStage] = [
            CurriculumStage(
                name="lift-near-object",
                progress_upper=0.20,
                object_xy_range=0.06,
                goal_radius_min=0.00,
                goal_radius_max=0.04,
                goal_min_z=0.04,
                goal_max_z=0.08,
                air_goal_prob=1.00,
                distance_threshold=0.08,
                shaping_scale=1.00,
                spawn_ee_above_object=True,
                ee_hover_height=0.10,
                settle_steps=2,
            ),
            CurriculumStage(
                name="short-carry-mixed",
                progress_upper=0.45,
                object_xy_range=0.08,
                goal_radius_min=0.02,
                goal_radius_max=0.10,
                goal_min_z=0.00,
                goal_max_z=0.10,
                air_goal_prob=0.70,
                distance_threshold=0.07,
                shaping_scale=0.70,
                spawn_ee_above_object=True,
                ee_hover_height=0.09,
                settle_steps=1,
            ),
            CurriculumStage(
                name="medium-carry",
                progress_upper=0.75,
                object_xy_range=0.14,
                goal_radius_min=0.03,
                goal_radius_max=0.16,
                goal_min_z=0.00,
                goal_max_z=0.14,
                air_goal_prob=0.70,
                distance_threshold=0.06,
                shaping_scale=0.30,
                spawn_ee_above_object=True,
                ee_hover_height=0.08,
                settle_steps=1,
            ),
            CurriculumStage(
                name="full-task",
                progress_upper=1.01,
                object_xy_range=self.base_obj_xy_range,
                goal_radius_min=0.00,
                goal_radius_max=self.base_goal_xy_range,
                goal_min_z=0.00,
                goal_max_z=self.base_goal_z_range,
                air_goal_prob=0.70,
                distance_threshold=self.base_distance_threshold,
                shaping_scale=0.00,
                spawn_ee_above_object=False,
                ee_hover_height=0.00,
                use_env_default_goal=True,
                settle_steps=0,
            ),
        ]

    def _progress(self) -> float:
        return min(self.local_env_steps / float(self.total_env_steps_target), 1.0)

    def _pick_stage(self) -> CurriculumStage:
        p = self._progress()
        for stage in self.stages:
            if p <= stage.progress_upper:
                return stage
        return self.stages[-1]

    def _set_square_range(self, center: np.ndarray, xy_range: float, z_low: float, z_high: float):
        low = np.array(center, dtype=np.float64).copy()
        high = np.array(center, dtype=np.float64).copy()
        low[:2] += np.array([-xy_range / 2.0, -xy_range / 2.0])
        high[:2] += np.array([xy_range / 2.0, xy_range / 2.0])
        low[2] = z_low
        high[2] = z_high
        return low, high

    def _apply_stage_pre_reset(self, stage: CurriculumStage) -> None:
        base = self.unwrapped
        base.obj_xy_range = stage.object_xy_range
        base.distance_threshold = stage.distance_threshold
        base.goal_z_range = stage.goal_max_z

        base.obj_range_low, base.obj_range_high = self._set_square_range(
            self.base_obj_center, stage.object_xy_range, self.base_obj_center[2], self.base_obj_center[2]
        )
        base.goal_range_low, base.goal_range_high = self._set_square_range(
            self.base_goal_center,
            max(stage.goal_radius_max * 2.0, 1e-3),
            self.base_goal_center[2],
            self.base_goal_center[2] + stage.goal_max_z,
        )

    def _sample_goal_near_object(self, object_pos: np.ndarray, stage: CurriculumStage) -> np.ndarray:
        base = self.unwrapped
        goal = np.array(object_pos, dtype=np.float64).copy()

        radius = float(self.np_random.uniform(stage.goal_radius_min, stage.goal_radius_max))
        theta = float(self.np_random.uniform(-np.pi, np.pi))
        goal[:2] += radius * np.array([np.cos(theta), np.sin(theta)])

        if self.np_random.random() < stage.air_goal_prob:
            goal[2] = float(base.initial_object_height + self.np_random.uniform(stage.goal_min_z, stage.goal_max_z))
        else:
            goal[2] = float(base.initial_object_height)

        goal = np.clip(goal, self.base_goal_range_low, self.base_goal_range_high)
        return goal.astype(np.float32)

    @property
    def np_random(self):
        return self.unwrapped.np_random

    def _move_ee_above_object(self, object_pos: np.ndarray, stage: CurriculumStage) -> None:
        base = self.unwrapped
        if not hasattr(base, "set_mocap_pose"):
            return

        hover = np.array(object_pos, dtype=np.float64).copy()
        hover[2] = max(float(object_pos[2] + stage.ee_hover_height), float(base.initial_object_height + 0.08))
        base.set_mocap_pose(hover, base.grasp_site_pose)

        if hasattr(base, "ctrl_range") and getattr(base, "block_gripper", True) is False:
            open_half_width = float(np.clip(0.04, base.ctrl_range[-1, 0], base.ctrl_range[-1, 1]))
            base.data.ctrl[-2:] = open_half_width

        for _ in range(max(stage.settle_steps, 0)):
            base._mujoco_step()
        base._mujoco.mj_forward(base.model, base.data)

    def _refresh_obs(self) -> Dict[str, np.ndarray]:
        obs = self.unwrapped._get_obs().copy()
        return obs

    def reset(self, **kwargs):
        self.current_stage = self._pick_stage()
        self.episode_idx += 1
        self._apply_stage_pre_reset(self.current_stage)

        obs, info = self.env.reset(**kwargs)
        object_pos = np.array(obs["achieved_goal"], dtype=np.float64).copy()

        if self.current_stage.spawn_ee_above_object:
            self._move_ee_above_object(object_pos, self.current_stage)

        if not self.current_stage.use_env_default_goal:
            new_goal = self._sample_goal_near_object(object_pos, self.current_stage)
            self.unwrapped.goal = new_goal.copy()
            if hasattr(self.unwrapped, "_render_callback"):
                self.unwrapped._render_callback()

        obs = self._refresh_obs()
        info = dict(info)
        info["curriculum_stage"] = self.current_stage.name
        info["curriculum_progress"] = self._progress()
        return obs, info

    def _compute_shaping_bonus(self, obs: Dict[str, np.ndarray]) -> float:
        if self.current_stage is None or self.current_stage.shaping_scale <= 0.0:
            return 0.0

        ee_position = np.array(obs["observation"][:3], dtype=np.float64)
        object_position = np.array(obs["achieved_goal"], dtype=np.float64)
        goal_position = np.array(obs["desired_goal"], dtype=np.float64)
        fingers_width = float(obs["observation"][6]) if obs["observation"].shape[0] > 6 else 0.0

        reach_dist = float(np.linalg.norm(ee_position - object_position))
        goal_dist = float(np.linalg.norm(object_position - goal_position))
        lift_height = max(float(object_position[2] - self.unwrapped.initial_object_height), 0.0)

        reach_term = np.exp(-12.0 * reach_dist)
        goal_term = np.exp(-8.0 * goal_dist)
        lift_term = np.clip(lift_height / 0.10, 0.0, 1.0)
        grasp_hint = 1.0 if (reach_dist < 0.035 and fingers_width < 0.06) else 0.0

        bonus = (
            0.35 * reach_term
            + 0.55 * goal_term
            + 0.35 * lift_term
            + 0.15 * grasp_hint
        )
        return float(self.current_stage.shaping_scale * bonus)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.local_env_steps += 1

        info = dict(info)
        if self.current_stage is not None:
            info["curriculum_stage"] = self.current_stage.name
            info["curriculum_progress"] = self._progress()

        # sparse pick-and-place 에만 쓰는 보너스. 마지막 stage 에서는 자동으로 0.
        shaping_bonus = self._compute_shaping_bonus(obs)
        reward = float(reward) + shaping_bonus

        if info.get("is_success", False):
            reward += 1.0

        info["curriculum_shaping_bonus"] = shaping_bonus
        return obs, reward, terminated, truncated, info
