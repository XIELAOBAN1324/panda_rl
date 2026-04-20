import os

import numpy as np

from panda_mujoco_gym.envs.panda_env import FrankaEnv


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "pick_and_place_window.xml")


class FrankaPickAndPlaceWindowEnv(FrankaEnv):
    def __init__(self, reward_type, **kwargs):
        self.is_window_task = True
        self.window_goal_height = 0.65
        self.window_ring_radius = 0.80
        self.window_sector_min_angle_deg = -60.0
        self.window_sector_max_angle_deg = 0.0
        self.window_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.window_goal_offset = 0.03
        self.window_position_threshold = 0.03
        self.orientation_threshold_cos = float(np.cos(np.deg2rad(20.0)))
        self.window_outer_size = np.array([0.40, 0.40], dtype=np.float64)
        self.window_opening_size = np.array([0.35, 0.35], dtype=np.float64)
        self.window_frame_width = 0.025
        self.window_plane_thickness = 0.025
        self.preinsert_offset = 0.06

        self.object_spawn_height = 0.12
        self.object_spawn_normal_range = (0.12, 0.24)
        self.object_spawn_lateral_range = 0.10
        self.default_object_offset = 0.18
        self.object_settle_steps = 120
        self.object_settle_linvel_threshold = 5e-3
        self.object_settle_angvel_threshold = 5e-2
        self.object_half_extents = np.array([0.02, 0.02, 0.02], dtype=np.float64)

        self.grasp_attach_distance_threshold = 0.065
        self.grasp_attach_width_threshold = 0.05
        self.grasp_release_width_threshold = 0.06

        self.plane_constraint_eps = 1e-4
        self.max_spawn_sampling_attempts = 64
        self.safe_start_gap = 0.08
        self.safe_start_height = 0.28

        self.is_attached = False
        self.window_body_id = -1
        self.obj_body_id = -1
        self.ee_center_body_id = -1
        self.obj_joint_id = -1
        self.obj_qpos_adr = -1
        self.obj_dof_adr = -1
        self.grasp_weld_eq_id = -1
        self.link0_body_id = -1
        self.hand_body_id = -1
        self.robot_body_ids: set[int] = set()
        self.hand_body_ids: set[int] = set()
        self.plane_robot_geom_ids: list[int] = []
        self.plane_object_geom_ids: list[int] = []
        self.plane_eef_geom_ids: list[int] = []
        self.mesh_vertices_by_mesh_id: dict[int, np.ndarray] = {}
        self.window_center = np.array([self.window_ring_radius, 0.0, self.window_goal_height], dtype=np.float64)
        self.window_safe_goal = self.window_center - self.window_normal * self.window_goal_offset
        self._last_plane_violation = False
        self._last_max_plane_penetration = 0.0

        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=False,
            distance_threshold=self.window_position_threshold,
            goal_xy_range=0.3,
            obj_xy_range=0.3,
            goal_x_offset=0.0,
            goal_z_range=0.0,
            orientation_action_size=3,
            position_action_scale=0.05,
            rotation_action_scale=0.20,
            **kwargs,
        )

        sector_min_angle = np.deg2rad(self.window_sector_min_angle_deg)
        sector_max_angle = np.deg2rad(self.window_sector_max_angle_deg)
        self.goal_range_low = np.array(
            [
                self.window_ring_radius * np.cos(sector_max_angle),
                self.window_ring_radius * np.sin(sector_min_angle),
                self.window_goal_height,
            ],
            dtype=np.float64,
        )
        self.goal_range_high = np.array(
            [
                self.window_ring_radius * np.cos(sector_min_angle),
                self.window_ring_radius * np.sin(sector_max_angle),
                self.window_goal_height,
            ],
            dtype=np.float64,
        )
        self.obj_range_low = np.array(
            [
                self.goal_range_low[0] - self.object_spawn_normal_range[1],
                self.goal_range_low[1] - self.object_spawn_lateral_range,
                0.0,
            ],
            dtype=np.float64,
        )
        self.obj_range_high = np.array(
            [
                self.goal_range_high[0] - self.object_spawn_normal_range[0],
                self.goal_range_high[1] + self.object_spawn_lateral_range,
                0.0,
            ],
            dtype=np.float64,
        )

    def _initialize_simulation(self) -> None:
        super()._initialize_simulation()
        self.window_body_id = self._model_names.body_name2id["window_frame"]
        self.obj_body_id = self._model_names.body_name2id["obj"]
        self.ee_center_body_id = self._model_names.body_name2id["ee_center_body"]
        self.obj_joint_id = self._model_names.joint_name2id["obj_joint"]
        self.obj_qpos_adr = int(self.model.jnt_qposadr[self.obj_joint_id])
        self.obj_dof_adr = int(self.model.jnt_dofadr[self.obj_joint_id])
        self.grasp_weld_eq_id = int(
            self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_EQUALITY, "obj_grasp_weld")
        )
        self.link0_body_id = self._model_names.body_name2id["link0"]
        self.hand_body_id = self._model_names.body_name2id["hand"]

        self.robot_body_ids = self._body_descendants(self.link0_body_id)
        self.hand_body_ids = self._body_descendants(self.hand_body_id)
        self.mesh_vertices_by_mesh_id = self._build_mesh_vertex_cache()
        self._build_plane_constraint_geom_sets()

        self._set_grasp_weld_active(False)
        self.window_center = self.model.body_pos[self.window_body_id].copy()
        self.window_safe_goal = self._compose_window_goal(self.window_center)[:3]
        self._mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self._last_plane_violation = False
        self._last_max_plane_penetration = 0.0
        pre_step_state = self._capture_sim_state()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._apply_pre_action_plane_clip()

        self._mujoco_step(action)
        self._step_callback()
        self._update_grasp_attachment(action)

        post_step_penetration = self._max_plane_penetration()
        if post_step_penetration > self.plane_constraint_eps:
            self._last_plane_violation = True
            self._last_max_plane_penetration = max(self._last_max_plane_penetration, post_step_penetration)
            self._restore_sim_state(pre_step_state)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        object_position = obs["achieved_goal"][:3]
        orientation_alignment = float(self._goal_alignment(obs["achieved_goal"], self.goal))
        plane_violation = bool(self._last_plane_violation)
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "ee_object_distance": float(np.linalg.norm(self.get_ee_position() - object_position)),
            "object_height": float(object_position[2] - self.initial_object_height),
            "orientation_alignment": orientation_alignment,
            "is_attached": bool(self.is_attached),
            "window_center": self.window_center.copy(),
            "window_safe_goal": self.window_safe_goal.copy(),
            "window_normal": self.goal[3:6].copy(),
            "position_error": float(self.goal_distance(obs["achieved_goal"], self.goal)),
            "plane_violation": plane_violation,
            "max_plane_penetration": float(self._last_max_plane_penetration),
        }
        terminated = bool(info["is_success"]) if self.terminate_on_success else False
        truncated = bool(self.compute_truncated(obs["achieved_goal"], self.goal, info))
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        position_distance = np.asarray(self.goal_distance(achieved_goal, desired_goal), dtype=np.float32)
        orientation_alignment = np.asarray(self._goal_alignment(achieved_goal, desired_goal), dtype=np.float32)
        orientation_error = 1.0 - np.clip(orientation_alignment, -1.0, 1.0)
        plane_violation = np.asarray(info.get("plane_violation", False), dtype=np.float32)
        success = (
            (position_distance < self.window_position_threshold)
            & (orientation_alignment >= self.orientation_threshold_cos)
            & (plane_violation <= 0.0)
        )
        if self.reward_type == "sparse":
            return -np.logical_not(success).astype(np.float32)

        reward = -position_distance - 0.25 * orientation_error
        ee_object_distance = float(info.get("ee_object_distance", 0.0))
        object_height = max(float(info.get("object_height", 0.0)), 0.0)

        reward -= 0.20 * ee_object_distance
        reward += 0.40 * min(object_height, 0.12)
        reward += 0.25 * (orientation_alignment >= self.orientation_threshold_cos).astype(np.float32)
        reward += 0.50 * success.astype(np.float32)
        reward -= 0.25 * plane_violation
        return reward

    def _get_obs(self) -> dict:
        base_obs = super()._get_obs()
        ee_forward_axis = self.get_ee_forward_axis().astype(np.float64)
        attached_flag = np.array([1.0 if self.is_attached else 0.0], dtype=np.float64)
        default_goal = self._compose_window_goal(self.window_center)
        desired_goal = self.goal.copy() if self.goal.shape == (6,) else default_goal
        achieved_goal = np.concatenate([base_obs["achieved_goal"], ee_forward_axis]).astype(np.float64)
        observation = np.concatenate([base_obs["observation"], ee_forward_axis, attached_flag]).astype(np.float64)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal.astype(np.float64),
        }

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        position_distance = self.goal_distance(achieved_goal, desired_goal)
        orientation_alignment = self._goal_alignment(achieved_goal, desired_goal)
        success = (
            (position_distance < self.window_position_threshold)
            and (orientation_alignment >= self.orientation_threshold_cos)
            and (not self._last_plane_violation)
        )
        return np.float32(success)

    def _render_callback(self) -> None:
        if self.goal.shape == (6,):
            self.window_center = self._window_center_from_goal(self.goal)
            self.window_safe_goal = self.goal[:3].copy()
        self.model.body_pos[self.window_body_id] = self.window_center
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        self._set_grasp_weld_active(False)

        open_half_width = float(np.clip(0.04, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1]))
        self.data.ctrl[-2:] = open_half_width

        self.window_center = self._sample_window_center()
        self.window_safe_goal = self._compose_window_goal(self.window_center)[:3]
        self.model.body_pos[self.window_body_id] = self.window_center

        self._sample_object()
        self._mujoco.mj_forward(self.model, self.data)
        self._settle_object()
        if not self._object_is_robot_side():
            self._sample_object()
            self._mujoco.mj_forward(self.model, self.data)
            self._settle_object()

        self._move_robot_to_safe_start()
        return True

    def _sample_goal(self) -> np.ndarray:
        self.window_safe_goal = self._compose_window_goal(self.window_center)[:3]
        return self._compose_window_goal(self.window_center)

    def _sample_window_center(self) -> np.ndarray:
        theta = float(
            self.np_random.uniform(
                np.deg2rad(self.window_sector_min_angle_deg),
                np.deg2rad(self.window_sector_max_angle_deg),
            )
        )
        return np.array(
            [
                self.window_ring_radius * np.cos(theta),
                self.window_ring_radius * np.sin(theta),
                self.window_goal_height,
            ],
            dtype=np.float64,
        )

    def _compose_window_goal(self, window_center: np.ndarray) -> np.ndarray:
        window_center = np.asarray(window_center, dtype=np.float64)
        safe_goal = window_center - self.window_normal * self.window_goal_offset
        return np.concatenate([safe_goal, self.window_normal.copy()]).astype(np.float64)

    def _window_center_from_goal(self, goal: np.ndarray) -> np.ndarray:
        goal = np.asarray(goal, dtype=np.float64)
        return goal[:3] + goal[3:6] * self.window_goal_offset

    def _sample_object(self) -> None:
        fallback_position = np.array(
            [
                self.window_center[0] - self.default_object_offset,
                self.window_center[1],
                self.object_spawn_height,
            ],
            dtype=np.float64,
        )
        fallback_position[0] = min(fallback_position[0], self.window_center[0] - self.window_goal_offset - 0.08)

        object_position = None
        for _ in range(self.max_spawn_sampling_attempts):
            normal_offset = float(self.np_random.uniform(*self.object_spawn_normal_range))
            lateral_offset = float(
                self.np_random.uniform(-self.object_spawn_lateral_range, self.object_spawn_lateral_range)
            )
            candidate = np.array(
                [
                    self.window_center[0] - normal_offset,
                    self.window_center[1] + lateral_offset,
                    self.object_spawn_height,
                ],
                dtype=np.float64,
            )
            if self._is_valid_object_spawn(candidate):
                object_position = candidate
                break

        if object_position is None:
            object_position = fallback_position

        object_xpos = np.concatenate([object_position, np.array([1.0, 0.0, 0.0, 0.0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)
        self.data.qvel[self.obj_dof_adr:self.obj_dof_adr + 6] = 0.0

    def _is_valid_object_spawn(self, object_position: np.ndarray) -> bool:
        object_position = np.asarray(object_position, dtype=np.float64)
        if object_position[0] >= (self.window_center[0] - self.window_goal_offset - 0.04):
            return False
        if object_position[0] <= 0.18:
            return False
        if abs(float(object_position[1])) > 0.40:
            return False
        return True

    def goal_distance(self, goal_a, goal_b):
        goal_a = np.asarray(goal_a, dtype=np.float64)
        goal_b = np.asarray(goal_b, dtype=np.float64)
        return np.linalg.norm(goal_a[..., :3] - goal_b[..., :3], axis=-1)

    def minimum_goal_object_distance(self) -> float:
        if self.reward_type != "sparse":
            return 0.0
        return float(self.window_position_threshold + 0.02)

    def get_window_center(self) -> np.ndarray:
        return np.asarray(self.window_center, dtype=np.float64).copy()

    def get_window_safe_goal(self) -> np.ndarray:
        return np.asarray(self.window_safe_goal, dtype=np.float64).copy()

    def get_window_normal(self) -> np.ndarray:
        return np.asarray(self.goal[3:6] if self.goal.shape == (6,) else self.window_normal, dtype=np.float64).copy()

    def _goal_alignment(self, achieved_goal, desired_goal) -> np.ndarray:
        achieved_axis = np.asarray(achieved_goal, dtype=np.float64)[..., 3:6]
        desired_axis = np.asarray(desired_goal, dtype=np.float64)[..., 3:6]
        achieved_axis = self._normalize_goal_axis(achieved_axis)
        desired_axis = self._normalize_goal_axis(desired_axis)
        return np.sum(achieved_axis * desired_axis, axis=-1)

    def _normalize_goal_axis(self, axis: np.ndarray) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        safe_norm = np.maximum(norm, 1e-8)
        return axis / safe_norm

    def _settle_object(self) -> None:
        settled_steps = 0
        for _ in range(self.object_settle_steps):
            self._mujoco_step()
            linear_speed = float(np.linalg.norm(self._utils.get_site_xvelp(self.model, self.data, "obj_site")))
            angular_speed = float(np.linalg.norm(self._utils.get_site_xvelr(self.model, self.data, "obj_site")))
            if (
                linear_speed < self.object_settle_linvel_threshold
                and angular_speed < self.object_settle_angvel_threshold
            ):
                settled_steps += 1
                if settled_steps >= 10:
                    break
            else:
                settled_steps = 0

        self.data.qvel[self.obj_dof_adr:self.obj_dof_adr + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        self.initial_object_height = float(self.get_object_position()[2])

    def _move_robot_to_safe_start(self) -> None:
        object_position = self.get_object_position().copy()
        start_position = np.array(
            [
                min(float(object_position[0] - self.safe_start_gap), float(self.window_center[0] - self.object_spawn_normal_range[1] - 0.02)),
                float(np.clip(object_position[1], -0.20, 0.20)),
                float(max(self.safe_start_height, self.initial_mocap_position[2])),
            ],
            dtype=np.float64,
        )

        for _ in range(10):
            self.set_mocap_pose(start_position, self.grasp_site_pose)
            self._mujoco_step()
            self._mujoco.mj_forward(self.model, self.data)
            penetration = self._max_plane_penetration()
            if penetration <= self.plane_constraint_eps:
                break
            start_position[0] -= 0.03

    def _update_grasp_attachment(self, action: np.ndarray) -> None:
        gripper_signal = float(action[-1])
        fingers_width = float(self.get_fingers_width())
        ee_object_distance = float(np.linalg.norm(self.get_ee_position() - self.get_object_position()))

        if self.is_attached:
            if fingers_width > self.grasp_release_width_threshold or gripper_signal > 0.2:
                self._set_grasp_weld_active(False)
            return

        if fingers_width < self.grasp_attach_width_threshold and ee_object_distance < self.grasp_attach_distance_threshold:
            self._activate_grasp_weld()

    def _activate_grasp_weld(self) -> None:
        ee_pos = self.data.xpos[self.ee_center_body_id].copy()
        ee_quat = self.data.xquat[self.ee_center_body_id].copy()
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        obj_quat = self.data.xquat[self.obj_body_id].copy()
        ee_rot = self.data.xmat[self.ee_center_body_id].reshape(3, 3)
        rel_pos = ee_rot.T @ (obj_pos - ee_pos)
        rel_quat = self._quat_multiply(self._quat_conjugate(ee_quat), obj_quat)

        self.model.eq_data[self.grasp_weld_eq_id, :3] = 0.0
        self.model.eq_data[self.grasp_weld_eq_id, 3:6] = rel_pos
        self.model.eq_data[self.grasp_weld_eq_id, 6:10] = self._normalize_vector(rel_quat)
        self.model.eq_data[self.grasp_weld_eq_id, 10] = 1.0
        self._set_grasp_weld_active(True)

    def _set_grasp_weld_active(self, active: bool) -> None:
        if self.grasp_weld_eq_id < 0:
            return
        self.model.eq_active[self.grasp_weld_eq_id] = np.uint8(1 if active else 0)
        self.is_attached = bool(active)
        self.data.qvel[self.obj_dof_adr:self.obj_dof_adr + 6] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def _capture_sim_state(self) -> dict:
        return {
            "time": float(self.data.time),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "act": None if self.model.na == 0 else self.data.act.copy(),
            "mocap_pos": self.data.mocap_pos.copy(),
            "mocap_quat": self.data.mocap_quat.copy(),
            "eq_active": self.model.eq_active.copy(),
            "eq_data": self.model.eq_data.copy(),
            "is_attached": bool(self.is_attached),
        }

    def _restore_sim_state(self, state: dict) -> None:
        self.data.time = float(state["time"])
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.ctrl[:] = state["ctrl"]
        if self.model.na != 0 and state["act"] is not None:
            self.data.act[:] = state["act"]
        self.data.mocap_pos[:] = state["mocap_pos"]
        self.data.mocap_quat[:] = state["mocap_quat"]
        self.model.eq_active[:] = state["eq_active"]
        self.model.eq_data[:] = state["eq_data"]
        self.is_attached = bool(state["is_attached"])
        self._mujoco.mj_forward(self.model, self.data)

    def _apply_pre_action_plane_clip(self) -> None:
        if getattr(self.data, "mocap_pos", None) is None or self.data.mocap_pos.size == 0:
            return

        requested_x = float(self.data.mocap_pos[0, 0])
        eef_margin = self._current_eef_plane_margin()
        clip_x = float(self.window_center[0] - eef_margin - self.plane_constraint_eps)
        if requested_x > clip_x:
            self.data.mocap_pos[0, 0] = clip_x
            self._last_plane_violation = True
            self._last_max_plane_penetration = max(self._last_max_plane_penetration, requested_x - clip_x)

    def _current_eef_plane_margin(self) -> float:
        if not self.plane_eef_geom_ids:
            return 0.0
        ee_x = float(self.get_ee_position()[0])
        max_support_x = max(self._geom_support_on_normal(geom_id) for geom_id in self.plane_eef_geom_ids)
        return max(0.0, max_support_x - ee_x)

    def _max_plane_penetration(self) -> float:
        if not self.plane_robot_geom_ids and not self.plane_object_geom_ids:
            return 0.0
        plane_offset = float(np.dot(self.window_center, self.window_normal))
        max_penetration = float("-inf")
        for geom_id in self.plane_robot_geom_ids + self.plane_object_geom_ids:
            support = self._geom_support_on_normal(geom_id)
            max_penetration = max(max_penetration, float(support - plane_offset))
        return max(0.0, max_penetration)

    def _object_is_robot_side(self) -> bool:
        return self._max_object_penetration() <= self.plane_constraint_eps

    def _max_object_penetration(self) -> float:
        if not self.plane_object_geom_ids:
            return 0.0
        plane_offset = float(np.dot(self.window_center, self.window_normal))
        max_penetration = float("-inf")
        for geom_id in self.plane_object_geom_ids:
            support = self._geom_support_on_normal(geom_id)
            max_penetration = max(max_penetration, float(support - plane_offset))
        return max(0.0, max_penetration)

    def _geom_support_on_normal(self, geom_id: int) -> float:
        normal = self.window_normal
        geom_pos = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(self.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        local_direction = geom_rot.T @ normal
        geom_type = int(self.model.geom_type[geom_id])
        geom_size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        center_projection = float(np.dot(geom_pos, normal))

        geom_enum = self._mujoco.mjtGeom
        if geom_type == int(geom_enum.mjGEOM_BOX):
            support_radius = float(np.dot(np.abs(local_direction[:3]), geom_size[:3]))
            return center_projection + support_radius
        if geom_type == int(geom_enum.mjGEOM_SPHERE):
            return center_projection + float(geom_size[0])
        if geom_type in (int(geom_enum.mjGEOM_CAPSULE), int(geom_enum.mjGEOM_CYLINDER)):
            radial = float(np.linalg.norm(local_direction[:2]))
            support_radius = float(geom_size[0] * radial + geom_size[1] * abs(local_direction[2]))
            return center_projection + support_radius
        if geom_type == int(geom_enum.mjGEOM_ELLIPSOID):
            support_radius = float(np.linalg.norm(geom_size[:3] * local_direction[:3]))
            return center_projection + support_radius
        if geom_type == int(geom_enum.mjGEOM_MESH):
            mesh_id = int(self.model.geom_dataid[geom_id])
            vertices = self.mesh_vertices_by_mesh_id.get(mesh_id)
            if vertices is not None and vertices.size > 0:
                support_radius = float(np.max(vertices @ local_direction))
                return center_projection + support_radius
            return center_projection + float(self.model.geom_rbound[geom_id])
        return center_projection + float(self.model.geom_rbound[geom_id])

    def _build_mesh_vertex_cache(self) -> dict[int, np.ndarray]:
        cache: dict[int, np.ndarray] = {}
        for mesh_id in range(int(self.model.nmesh)):
            vert_adr = int(self.model.mesh_vertadr[mesh_id])
            vert_num = int(self.model.mesh_vertnum[mesh_id])
            if vert_num <= 0:
                cache[mesh_id] = np.zeros((0, 3), dtype=np.float64)
                continue
            cache[mesh_id] = np.asarray(
                self.model.mesh_vert[vert_adr:vert_adr + vert_num],
                dtype=np.float64,
            ).copy()
        return cache

    def _build_plane_constraint_geom_sets(self) -> None:
        self.plane_robot_geom_ids = []
        self.plane_object_geom_ids = []
        self.plane_eef_geom_ids = []
        for geom_id in range(int(self.model.ngeom)):
            body_id = int(self.model.geom_bodyid[geom_id])
            if not self._is_collision_geom(geom_id):
                continue
            if body_id in self.robot_body_ids:
                self.plane_robot_geom_ids.append(geom_id)
                if body_id in self.hand_body_ids:
                    self.plane_eef_geom_ids.append(geom_id)
            elif body_id == self.obj_body_id:
                self.plane_object_geom_ids.append(geom_id)

    def _is_collision_geom(self, geom_id: int) -> bool:
        contype = int(self.model.geom_contype[geom_id])
        conaffinity = int(self.model.geom_conaffinity[geom_id])
        group = int(self.model.geom_group[geom_id])
        return bool(contype or conaffinity or group == 3)

    def _body_descendants(self, root_body_id: int) -> set[int]:
        descendants = set()
        for body_id in range(int(self.model.nbody)):
            current_id = body_id
            while current_id >= 0:
                if current_id == root_body_id:
                    descendants.add(body_id)
                    break
                parent_id = int(self.model.body_parentid[current_id])
                if parent_id == current_id:
                    break
                current_id = parent_id
        descendants.add(root_body_id)
        return descendants

    def _quat_conjugate(self, quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    def _quat_multiply(self, quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
        result = np.empty(4, dtype=np.float64)
        self._mujoco.mju_mulQuat(result, quat_a, quat_b)
        return result
