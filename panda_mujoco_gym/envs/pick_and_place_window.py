import os

import numpy as np

from panda_mujoco_gym.envs.insert_reward import compute_insert_dense_reward
from panda_mujoco_gym.envs.panda_env import FrankaEnv


MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), "../assets/", "pick_and_place_window.xml")


class FrankaPickAndPlaceWindowEnv(FrankaEnv):
    def __init__(self, reward_type, **kwargs):
        self.target_visible_in_rgb_array = bool(kwargs.pop("target_visible_in_rgb_array", True))
        self.reset_mode = str(kwargs.pop("reset_mode", "post_grasp_lifted"))
        if self.reset_mode not in {"full_task", "post_grasp_lifted", "post_grasp_prealign"}:
            raise ValueError(f"Unsupported reset_mode: {self.reset_mode}")
        self.is_insertion_task = self.reset_mode in {"post_grasp_lifted", "post_grasp_prealign"}
        self.is_prealign_task = self.reset_mode == "post_grasp_prealign"
        self.scripted_pickup = bool(kwargs.pop("scripted_pickup", True))
        self.constrain_robot_to_window_plane = bool(
            kwargs.pop("constrain_robot_to_window_plane", not self.is_insertion_task)
        )

        self.success_position_threshold = float(kwargs.pop("success_position_threshold", 0.005))
        self.success_orientation_deg = float(kwargs.pop("success_orientation_deg", 5.0))
        self.collision_termination = bool(kwargs.pop("collision_termination", True))
        self.post_grasp_jitter_xy = float(kwargs.pop("post_grasp_jitter_xy", 0.02))
        self.post_grasp_jitter_z = float(kwargs.pop("post_grasp_jitter_z", 0.01))
        self.post_grasp_jitter_rot_deg = float(kwargs.pop("post_grasp_jitter_rot_deg", 5.0))
        glass_fit_orientation_deg = float(
            kwargs.pop(
                "glass_fit_orientation_deg",
                self.success_orientation_deg if self.is_insertion_task else 12.0,
            )
        )
        glass_plane_tolerance = kwargs.pop("glass_plane_tolerance", None)
        if glass_plane_tolerance is None:
            glass_plane_tolerance = self.success_position_threshold if self.is_insertion_task else 0.01

        self.is_window_task = True
        self.window_goal_height = 0.65
        self.window_ring_radius = 0.80
        self.window_sector_min_angle_deg = -60.0
        self.window_sector_max_angle_deg = 0.0
        self.window_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.window_goal_offset = 0.03
        self.window_position_threshold = self.success_position_threshold
        self.orientation_threshold_cos = float(np.cos(np.deg2rad(self.success_orientation_deg)))
        self.window_outer_size = np.array([0.66, 0.84], dtype=np.float64)
        self.window_opening_size = np.array([0.36, 0.54], dtype=np.float64)
        self.window_frame_width = 0.15
        self.window_plane_thickness = 0.025
        self.preinsert_offset = 0.06
        self.glass_thickness = 0.004
        self.glass_size_xy = self.window_opening_size - 0.01
        self.glass_half_extents = np.array(
            [self.glass_size_xy[0] / 2.0, self.glass_size_xy[1] / 2.0, self.glass_thickness / 2.0],
            dtype=np.float64,
        )
        self.glass_local_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.glass_plane_tolerance = float(glass_plane_tolerance)
        self.glass_alignment_threshold_cos = float(np.cos(np.deg2rad(glass_fit_orientation_deg)))
        self.glass_center_region_xy = np.array([0.06, 0.06], dtype=np.float64)
        self.target_glass_normal = self.window_normal.copy()

        self.pickup_station_center = np.array([0.0, -0.60, 0.25], dtype=np.float64)
        self.pickup_station_size = np.array([0.49, 0.63, 0.248], dtype=np.float64)
        self.pickup_station_pad_height = 0.248
        self.object_settle_steps = 120
        self.object_settle_linvel_threshold = 5e-3
        self.object_settle_angvel_threshold = 5e-2
        self.object_half_extents = self.glass_half_extents.copy()

        self.grasp_attach_distance_threshold = 0.020
        self.suction_attach_signal_threshold = -0.2
        self.suction_release_signal_threshold = 0.2
        self.suction_attach_alignment_threshold_cos = float(np.cos(np.deg2rad(18.0)))

        self.plane_constraint_eps = 1e-4
        self.max_spawn_sampling_attempts = 1
        self.safe_start_gap = 0.08
        self.safe_start_height = 0.28
        self.post_grasp_canonical_goal_margin = 0.14
        self.post_grasp_min_lift_above_object = 0.18
        self.post_grasp_jitter_attempts = 16
        self.post_grasp_motion_steps = 12
        self.post_prealign_jitter_xy = 0.01
        self.post_prealign_jitter_z = 0.005
        self.post_prealign_jitter_rot_deg = 4.0
        self.post_prealign_offset = 0.14
        self.post_prealign_height = 0.12
        self.post_prealign_transport_height = 0.14

        self.is_attached = False
        self._grasp_relative_position = np.zeros(3, dtype=np.float64)
        self._grasp_relative_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
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
        self.window_geom_ids: list[int] = []
        self.object_collision_geom_ids: list[int] = []
        self.eef_collision_geom_ids: list[int] = []
        self.mesh_vertices_by_mesh_id: dict[int, np.ndarray] = {}
        self.window_center = np.array([self.window_ring_radius, 0.0, self.window_goal_height], dtype=np.float64)
        self.window_safe_goal = self.window_center - self.window_normal * self.window_goal_offset
        self._last_plane_violation = False
        self._last_max_plane_penetration = 0.0
        self._last_collision = False
        self._last_collision_with_window = False
        self.target_site_id = -1

        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            reward_type=reward_type,
            block_gripper=False,
            include_gripper_action=not self.scripted_pickup,
            fixed_gripper_action=-1.0 if self.scripted_pickup else 0.0,
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
                self.pickup_station_center[0],
                self.pickup_station_center[1],
                0.0,
            ],
            dtype=np.float64,
        )
        self.obj_range_high = np.array(
            [
                self.pickup_station_center[0],
                self.pickup_station_center[1],
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
        self.target_site_id = self._model_names.site_name2id["target"]
        self.link0_body_id = self._model_names.body_name2id["link0"]
        self.hand_body_id = self._model_names.body_name2id["hand"]

        self.robot_body_ids = self._body_descendants(self.link0_body_id)
        self.hand_body_ids = self._body_descendants(self.hand_body_id)
        self.mesh_vertices_by_mesh_id = self._build_mesh_vertex_cache()
        self._build_collision_geom_sets()

        self._set_grasp_weld_active(False)
        self.window_center = self.model.body_pos[self.window_body_id].copy()
        self.window_safe_goal = self._compose_window_goal(self.window_center)[:3]
        self._set_target_visibility()
        self._mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        self._last_plane_violation = False
        self._last_max_plane_penetration = 0.0
        self._last_collision = False
        self._last_collision_with_window = False
        pre_step_state = self._capture_sim_state()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._apply_pre_action_plane_clip()

        self._mujoco_step(action)
        self._step_callback()
        if self.scripted_pickup:
            if not self.is_attached:
                self._activate_grasp_weld()
        else:
            self._update_grasp_attachment(action)
        self._last_collision_with_window = self._has_window_collision()

        object_penetration = self._max_object_penetration()
        robot_penetration = self._max_robot_plane_penetration() if self.constrain_robot_to_window_plane else 0.0
        post_step_penetration = max(object_penetration, robot_penetration)
        if post_step_penetration > self.plane_constraint_eps:
            self._last_plane_violation = True
            self._last_max_plane_penetration = max(self._last_max_plane_penetration, post_step_penetration)
            self._restore_sim_state(pre_step_state)

        self._last_collision = bool(self._last_plane_violation or self._last_collision_with_window)

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        object_position = self.get_object_position()
        orientation_alignment = float(self._goal_alignment(obs["achieved_goal"], self.goal))
        inplane_alignment = float(self._glass_inplane_alignment())
        pose_alignment = float(min(orientation_alignment, inplane_alignment))
        plane_violation = bool(self._last_plane_violation)
        collision_with_window = bool(self._last_collision_with_window)
        collision = bool(self._last_collision)
        glass_fits_window = bool(self._glass_fits_window())
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "ee_object_distance": float(np.linalg.norm(self.get_ee_position() - object_position)),
            "object_height": float(object_position[2] - self.initial_object_height),
            "orientation_alignment": orientation_alignment,
            "inplane_alignment": inplane_alignment,
            "pose_alignment": pose_alignment,
            "is_attached": bool(self.is_attached),
            "glass_fits_window": glass_fits_window,
            "window_center": self.window_center.copy(),
            "window_safe_goal": self.window_safe_goal.copy(),
            "window_normal": self.goal[3:6].copy(),
            "position_error": float(self.goal_distance(obs["achieved_goal"], self.goal)),
            "plane_violation": plane_violation,
            "max_plane_penetration": float(self._last_max_plane_penetration),
            "collision": collision,
            "collision_with_window": collision_with_window,
        }

        terminated = bool(info["is_success"]) if self.terminate_on_success else False
        if self.collision_termination and collision:
            terminated = True
        truncated = False if terminated else bool(self.compute_truncated(obs["achieved_goal"], self.goal, info))
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        position_distance = np.asarray(self.goal_distance(achieved_goal, desired_goal), dtype=np.float32)
        orientation_alignment = np.asarray(self._goal_alignment(achieved_goal, desired_goal), dtype=np.float32)
        orientation_error = 1.0 - np.clip(orientation_alignment, -1.0, 1.0)
        inplane_alignment = np.asarray(info.get("inplane_alignment", orientation_alignment), dtype=np.float32)
        inplane_error = 1.0 - np.clip(inplane_alignment, -1.0, 1.0)
        plane_violation = np.asarray(info.get("plane_violation", False), dtype=np.float32)
        collision = np.asarray(info.get("collision", False), dtype=np.float32)
        plate_fits = np.asarray(info.get("glass_fits_window", False), dtype=np.float32)
        success = (
            (plate_fits > 0.0)
            & (position_distance < self.window_position_threshold)
            & (orientation_alignment >= self.orientation_threshold_cos)
            & (inplane_alignment >= self.orientation_threshold_cos)
            & (plane_violation <= 0.0)
            & (collision <= 0.0)
        )
        if self.reward_type == "sparse":
            return -np.logical_not(success).astype(np.float32)

        if self.is_insertion_task:
            return compute_insert_dense_reward(
                distances=position_distance,
                orientation_error=orientation_error,
                inplane_error=inplane_error,
                orientation_alignment=orientation_alignment,
                inplane_alignment=inplane_alignment,
                glass_fits_window=plate_fits,
                collision=collision,
                plane_violation=plane_violation,
                distance_threshold=self.window_position_threshold,
                orientation_threshold=self.orientation_threshold_cos,
            )

        reward = -position_distance - 0.35 * orientation_error
        ee_object_distance = float(info.get("ee_object_distance", 0.0))
        object_height = max(float(info.get("object_height", 0.0)), 0.0)
        attached_bonus = 0.20 if bool(info.get("is_attached", False)) else 0.0
        fit_bonus = 0.45 if bool(info.get("glass_fits_window", False)) else 0.0

        reward -= 0.20 * ee_object_distance
        reward += 0.40 * min(object_height, 0.12)
        reward += 0.25 * (orientation_alignment >= self.orientation_threshold_cos).astype(np.float32)
        reward += attached_bonus
        reward += fit_bonus
        reward += 0.50 * success.astype(np.float32)
        reward -= 0.25 * plane_violation
        reward -= 0.50 * collision
        return reward.astype(np.float32)

    def _get_obs(self) -> dict:
        ee_position = self.get_ee_position().copy()
        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt
        ee_forward_axis = self.get_ee_forward_axis().astype(np.float64)
        glass_tangent_axis = self.get_glass_tangent_axis().astype(np.float64)
        goal_position = self.goal[:3].copy() if self.goal.shape == (6,) else self.window_center.copy()
        goal_normal = self.get_window_normal()
        object_position = self.get_object_position().copy()
        relative_goal = goal_position - object_position
        attached_flag = np.array([1.0 if self.is_attached else 0.0], dtype=np.float64)
        default_goal = self._compose_window_goal(self.window_center)
        desired_goal = self.goal.copy() if self.goal.shape == (6,) else default_goal
        achieved_goal = np.concatenate([object_position, -ee_forward_axis]).astype(np.float64)
        observation = np.concatenate(
            [
                ee_position,
                ee_velocity,
                relative_goal,
                goal_normal,
                ee_forward_axis,
                glass_tangent_axis,
                attached_flag,
            ]
        ).astype(np.float64)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal.astype(np.float64),
        }

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        position_distance = self.goal_distance(achieved_goal, desired_goal)
        orientation_alignment = self._goal_alignment(achieved_goal, desired_goal)
        inplane_alignment = self._glass_inplane_alignment()
        glass_fits = self._glass_fits_window()
        collision_free = (not self._last_collision) if self.collision_termination else True
        success = (
            glass_fits
            and (position_distance < self.window_position_threshold)
            and (orientation_alignment >= self.orientation_threshold_cos)
            and (inplane_alignment >= self.orientation_threshold_cos)
            and (not self._last_plane_violation)
            and collision_free
        )
        return np.float32(success)

    def _render_callback(self) -> None:
        if self.goal.shape == (6,):
            self.window_center = self._window_center_from_goal(self.goal)
            self.window_safe_goal = self.goal[:3].copy()
        self.model.body_pos[self.window_body_id] = self.window_center
        self._set_target_visibility()
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        self._set_grasp_weld_active(False)
        self._last_collision = False
        self._last_collision_with_window = False
        self._last_plane_violation = False
        self._last_max_plane_penetration = 0.0

        self.window_center = self._sample_window_center()
        self.window_safe_goal = self._compose_window_goal(self.window_center)[:3]
        self.model.body_pos[self.window_body_id] = self.window_center
        self._set_target_visibility()

        self._sample_object()
        self._mujoco.mj_forward(self.model, self.data)
        self._settle_object()
        if not self._object_is_robot_side():
            self._sample_object()
            self._mujoco.mj_forward(self.model, self.data)
            self._settle_object()

        if self.is_prealign_task:
            self._reset_to_post_grasp_prealign_state()
        elif self.is_insertion_task or self.scripted_pickup:
            self._reset_to_post_grasp_lifted_state()
        else:
            self._move_robot_to_safe_start()
        return True

    def _sample_goal(self) -> np.ndarray:
        self.window_safe_goal = self.window_center.copy()
        return np.concatenate([self.window_center.copy(), self.target_glass_normal.copy()]).astype(np.float64)

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
        return np.concatenate([window_center.copy(), self.target_glass_normal.copy()]).astype(np.float64)

    def _window_center_from_goal(self, goal: np.ndarray) -> np.ndarray:
        goal = np.asarray(goal, dtype=np.float64)
        return goal[:3].copy()

    def _sample_object(self) -> None:
        object_position = self.pickup_station_center.copy()
        object_xpos = np.concatenate([object_position, np.array([1.0, 0.0, 0.0, 0.0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)
        self._zero_object_velocity()

    def _is_valid_object_spawn(self, object_position: np.ndarray) -> bool:
        object_position = np.asarray(object_position, dtype=np.float64)
        return np.allclose(object_position, self.pickup_station_center, atol=1e-6)

    def goal_distance(self, goal_a, goal_b):
        goal_a = np.asarray(goal_a, dtype=np.float64)
        goal_b = np.asarray(goal_b, dtype=np.float64)
        return np.linalg.norm(goal_a[..., :3] - goal_b[..., :3], axis=-1)

    def minimum_goal_object_distance(self) -> float:
        if self.reward_type != "sparse":
            return 0.0
        return float(self.window_position_threshold + 0.04)

    def get_window_center(self) -> np.ndarray:
        return np.asarray(self.window_center, dtype=np.float64).copy()

    def get_window_safe_goal(self) -> np.ndarray:
        return np.asarray(self.window_safe_goal, dtype=np.float64).copy()

    def get_window_normal(self) -> np.ndarray:
        return np.asarray(self.goal[3:6] if self.goal.shape == (6,) else self.target_glass_normal, dtype=np.float64).copy()

    def get_object_position(self) -> np.ndarray:
        if self.is_attached:
            ee_pos = self.data.xpos[self.ee_center_body_id].copy()
            ee_rot = self.data.xmat[self.ee_center_body_id].reshape(3, 3)
            return ee_pos + ee_rot @ self._grasp_relative_position
        return super().get_object_position()

    def get_object_rotation_matrix(self) -> np.ndarray:
        if self.is_attached:
            ee_rot = self.data.xmat[self.ee_center_body_id].reshape(3, 3)
            rel_rot = self._quat_to_mat(self._grasp_relative_quat)
            return (ee_rot @ rel_rot).astype(np.float64)
        return super().get_object_rotation_matrix()

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

        self._zero_object_velocity()
        self._mujoco.mj_forward(self.model, self.data)
        self.initial_object_height = float(self.get_object_position()[2])

    def _move_robot_to_safe_start(self) -> None:
        object_position = self.get_object_position().copy()
        start_position = np.array(
            [
                float(max(object_position[0] - self.safe_start_gap, 0.30)),
                float(np.clip(object_position[1], -0.05, 0.18)),
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

    def _reset_to_post_grasp_lifted_state(self) -> None:
        self._move_robot_to_safe_start()
        self._run_internal_pickup_routine()
        canonical_state = self._capture_sim_state()
        if not self._sample_post_grasp_jittered_state(canonical_state):
            self._restore_sim_state(canonical_state)
        self._stabilize_reset_state()

    def _reset_to_post_grasp_prealign_state(self) -> None:
        self._move_robot_to_safe_start()
        self._run_internal_pickup_routine()
        transport_position, prealign_position, prealign_quat = self._target_prealign_pose()
        self._move_attached_object_center(
            transport_position,
            self.get_mocap_quaternion().copy(),
            4 * self.post_grasp_motion_steps,
        )
        self._move_mocap_linearly(
            self.get_ee_position().copy(),
            prealign_quat,
            3 * self.post_grasp_motion_steps,
        )
        self._move_attached_object_center(prealign_position, prealign_quat, 5 * self.post_grasp_motion_steps)
        prealign_goal = self._compose_window_goal(self.window_center)
        for _ in range(4):
            position_error, orientation_alignment = self._glass_pose_error(prealign_goal)
            if position_error < 0.03 and orientation_alignment >= self.orientation_threshold_cos:
                break
            self._move_mocap_linearly(
                self.get_ee_position().copy(),
                prealign_quat,
                2 * self.post_grasp_motion_steps,
            )
            self._move_attached_object_center(
                prealign_position,
                prealign_quat,
                4 * self.post_grasp_motion_steps,
            )
        canonical_state = self._capture_sim_state()
        if not self._sample_post_prealign_jittered_state(canonical_state, prealign_position, prealign_quat):
            self._restore_sim_state(canonical_state)
        self._stabilize_reset_state()

    def _run_internal_pickup_routine(self) -> None:
        object_position = self.get_object_position().copy()
        approach_position = object_position + np.array([0.0, 0.0, 0.12], dtype=np.float64)
        descend_position = object_position + np.array([0.0, 0.0, 0.01], dtype=np.float64)

        self._move_mocap_linearly(approach_position, self.grasp_site_pose, self.post_grasp_motion_steps)
        self._move_mocap_linearly(descend_position, self.grasp_site_pose, self.post_grasp_motion_steps)

        attach_action = np.zeros(self.action_space.shape, dtype=np.float32)
        attach_action[-1] = -1.0
        self._update_grasp_attachment(attach_action)
        if not self.is_attached:
            self._activate_grasp_weld()

        lift_height = float(
            max(
                self.window_center[2] + self.post_grasp_canonical_goal_margin,
                self.initial_object_height + self.post_grasp_min_lift_above_object,
            )
        )
        canonical_object_position = np.array(
            [self.pickup_station_center[0], self.pickup_station_center[1], lift_height],
            dtype=np.float64,
        )
        self._move_attached_object_center(canonical_object_position, self.grasp_site_pose, 2 * self.post_grasp_motion_steps)

    def _sample_post_grasp_jittered_state(self, canonical_state: dict) -> bool:
        canonical_object_position = self.get_object_position().copy()
        canonical_mocap_quat = self.get_mocap_quaternion().copy()

        for _ in range(self.post_grasp_jitter_attempts):
            self._restore_sim_state(canonical_state)

            jittered_object_position = canonical_object_position.copy()
            jittered_object_position[0] += float(self.np_random.uniform(-self.post_grasp_jitter_xy, self.post_grasp_jitter_xy))
            jittered_object_position[1] += float(self.np_random.uniform(-self.post_grasp_jitter_xy, self.post_grasp_jitter_xy))
            jittered_object_position[2] += float(self.np_random.uniform(-self.post_grasp_jitter_z, self.post_grasp_jitter_z))

            angle = float(
                self.np_random.uniform(
                    -np.deg2rad(self.post_grasp_jitter_rot_deg),
                    np.deg2rad(self.post_grasp_jitter_rot_deg),
                )
            )
            axis = self._sample_random_axis()
            jitter_quat = self._quat_from_axis_angle(axis, angle)
            jittered_mocap_quat = self._normalize_vector(self._quat_multiply(jitter_quat, canonical_mocap_quat))

            self._move_mocap_linearly(self.get_ee_position().copy(), jittered_mocap_quat, self.post_grasp_motion_steps)
            self._move_attached_object_center(
                jittered_object_position,
                jittered_mocap_quat,
                self.post_grasp_motion_steps,
            )
            self._stabilize_reset_state()

            if self._is_valid_post_grasp_state():
                return True

        return False

    def _sample_post_prealign_jittered_state(
        self,
        canonical_state: dict,
        canonical_object_position: np.ndarray,
        canonical_mocap_quat: np.ndarray,
    ) -> bool:
        canonical_object_position = np.asarray(canonical_object_position, dtype=np.float64)
        canonical_mocap_quat = np.asarray(canonical_mocap_quat, dtype=np.float64)
        for _ in range(self.post_grasp_jitter_attempts):
            self._restore_sim_state(canonical_state)

            jittered_object_position = canonical_object_position.copy()
            jittered_object_position[0] += float(self.np_random.uniform(-self.post_prealign_jitter_xy, self.post_prealign_jitter_xy))
            jittered_object_position[1] += float(self.np_random.uniform(-self.post_prealign_jitter_xy, self.post_prealign_jitter_xy))
            jittered_object_position[2] += float(self.np_random.uniform(-self.post_prealign_jitter_z, self.post_prealign_jitter_z))

            angle = float(
                self.np_random.uniform(
                    -np.deg2rad(self.post_prealign_jitter_rot_deg),
                    np.deg2rad(self.post_prealign_jitter_rot_deg),
                )
            )
            axis = self._sample_random_axis()
            jitter_quat = self._quat_from_axis_angle(axis, angle)
            jittered_mocap_quat = self._normalize_vector(self._quat_multiply(jitter_quat, canonical_mocap_quat))

            self._move_mocap_linearly(self.get_ee_position().copy(), jittered_mocap_quat, self.post_grasp_motion_steps)
            self._move_attached_object_center(
                jittered_object_position,
                jittered_mocap_quat,
                self.post_grasp_motion_steps,
            )
            self._stabilize_reset_state()

            if self._is_valid_post_grasp_state():
                return True

        return False

    def _target_prealign_pose(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        retreat_position = self.window_center - self.window_normal * self.post_prealign_offset
        transport_position = retreat_position.copy()
        transport_position[2] = max(
            float(self.get_object_position()[2]),
            float(self.window_center[2] + self.post_prealign_transport_height),
        )
        desired_object_position = retreat_position + np.array([0.0, 0.0, self.post_prealign_height], dtype=np.float64)

        target_quat = self._quat_for_glass_normal(self.target_glass_normal)
        return (
            transport_position.astype(np.float64),
            desired_object_position.astype(np.float64),
            target_quat.astype(np.float64),
        )

    def _is_valid_post_grasp_state(self) -> bool:
        if not self.is_attached:
            return False
        if self._max_object_penetration() > self.plane_constraint_eps:
            return False
        if self._has_window_collision():
            return False
        return True

    def _stabilize_reset_state(self) -> None:
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def _move_mocap_linearly(
        self,
        target_position: np.ndarray,
        target_quat: np.ndarray,
        n_steps: int,
    ) -> None:
        start_position = self.get_ee_position().copy()
        target_position = np.asarray(target_position, dtype=np.float64)
        target_quat = np.asarray(target_quat, dtype=np.float64)

        for step_idx in range(max(int(n_steps), 1)):
            alpha = float(step_idx + 1) / float(max(int(n_steps), 1))
            position = (1.0 - alpha) * start_position + alpha * target_position
            self.set_mocap_pose(position, target_quat)
            self._mujoco_step()
            self._mujoco.mj_forward(self.model, self.data)

    def _move_attached_object_center(
        self,
        target_object_position: np.ndarray,
        target_mocap_quat: np.ndarray,
        n_steps: int,
    ) -> None:
        start_object_position = self.get_object_position().copy()
        target_object_position = np.asarray(target_object_position, dtype=np.float64)
        target_mocap_quat = np.asarray(target_mocap_quat, dtype=np.float64)

        for step_idx in range(max(int(n_steps), 1)):
            alpha = float(step_idx + 1) / float(max(int(n_steps), 1))
            desired_object_position = (1.0 - alpha) * start_object_position + alpha * target_object_position
            current_object_position = self.get_object_position().copy()
            current_ee_position = self.get_ee_position().copy()
            ee_target_position = current_ee_position + (desired_object_position - current_object_position)
            self.set_mocap_pose(ee_target_position, target_mocap_quat)
            self._mujoco_step()
            self._mujoco.mj_forward(self.model, self.data)

    def _update_grasp_attachment(self, action: np.ndarray) -> None:
        gripper_signal = float(action[-1])
        ee_object_distance = float(np.linalg.norm(self.get_ee_position() - self.get_object_position()))
        suction_alignment = float(np.dot(self.get_ee_forward_axis(), -self.get_glass_normal()))
        suction_center_offset = self._suction_center_offset_xy()

        if self.is_attached:
            if gripper_signal > self.suction_release_signal_threshold:
                self._set_grasp_weld_active(False)
            return

        if (
            gripper_signal < self.suction_attach_signal_threshold
            and ee_object_distance < self.grasp_attach_distance_threshold
            and suction_alignment >= self.suction_attach_alignment_threshold_cos
            and abs(float(suction_center_offset[0])) <= float(self.glass_center_region_xy[0])
            and abs(float(suction_center_offset[1])) <= float(self.glass_center_region_xy[1])
        ):
            self._activate_grasp_weld()

    def get_glass_normal(self) -> np.ndarray:
        obj_rot = self.get_object_rotation_matrix()
        return self._normalize_vector(obj_rot @ self.glass_local_normal)

    def get_glass_tangent_axis(self) -> np.ndarray:
        obj_rot = self.get_object_rotation_matrix()
        return self._normalize_vector(obj_rot[:, 0])

    def _target_glass_tangent_axis(self) -> np.ndarray:
        normal = self._normalize_vector(self.target_glass_normal)
        world_vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(world_vertical, normal))) > 0.95:
            world_vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent = world_vertical - float(np.dot(world_vertical, normal)) * normal
        return self._normalize_vector(tangent)

    def _glass_inplane_alignment(self) -> float:
        normal = self._normalize_vector(self.target_glass_normal)
        tangent = self.get_glass_tangent_axis()
        tangent = tangent - float(np.dot(tangent, normal)) * normal
        tangent = self._normalize_vector(tangent)
        target_tangent = self._target_glass_tangent_axis()
        # A 180-degree flip around the window normal is equivalent for the rectangle.
        return float(abs(np.clip(np.dot(tangent, target_tangent), -1.0, 1.0)))

    def _suction_center_offset_xy(self) -> np.ndarray:
        ee_pos = self.get_ee_position().copy()
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        obj_rot = self.data.xmat[self.obj_body_id].reshape(3, 3)
        relative = obj_rot.T @ (ee_pos - obj_pos)
        return relative[:2].astype(np.float64)

    def _glass_pose_error(self, desired_goal: np.ndarray) -> tuple[float, float]:
        desired_goal = np.asarray(desired_goal, dtype=np.float64)
        current_center = self.get_object_position().copy()
        current_normal = self.get_glass_normal()
        desired_center = desired_goal[:3]
        desired_normal = self._normalize_goal_axis(desired_goal[3:6])
        position_distance = float(np.linalg.norm(current_center - desired_center))
        orientation_alignment = float(abs(np.clip(np.dot(current_normal, desired_normal), -1.0, 1.0)))
        return position_distance, orientation_alignment

    def _glass_fits_window(self) -> bool:
        obj_pos = self.get_object_position().copy()
        obj_rot = self.get_object_rotation_matrix()
        local_corners = np.array(
            [
                [self.glass_half_extents[0], self.glass_half_extents[1], 0.0],
                [self.glass_half_extents[0], -self.glass_half_extents[1], 0.0],
                [-self.glass_half_extents[0], self.glass_half_extents[1], 0.0],
                [-self.glass_half_extents[0], -self.glass_half_extents[1], 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        world_points = (obj_rot @ local_corners.T).T + obj_pos
        plane_x = float(self.window_center[0])
        half_open_y = float(self.window_opening_size[1] / 2.0)
        half_open_z = float(self.window_opening_size[0] / 2.0)
        glass_normal = self.get_glass_normal()
        normal_alignment = float(abs(np.dot(glass_normal, self.target_glass_normal)))
        if normal_alignment < self.glass_alignment_threshold_cos:
            return False
        if bool(self._last_plane_violation):
            return False
        if self._max_object_penetration() > self.plane_constraint_eps:
            return False
        for point in world_points:
            if abs(float(point[0] - plane_x)) > self.glass_plane_tolerance:
                return False
            if abs(float(point[1] - self.window_center[1])) > half_open_y:
                return False
            if abs(float(point[2] - self.window_center[2])) > half_open_z:
                return False
        return True

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
        self._grasp_relative_position = rel_pos.copy()
        self._grasp_relative_quat = self._normalize_vector(rel_quat)
        self._set_grasp_weld_active(True)

    def _set_grasp_weld_active(self, active: bool) -> None:
        if self.grasp_weld_eq_id < 0:
            return
        self.model.eq_active[self.grasp_weld_eq_id] = np.uint8(1 if active else 0)
        self.is_attached = bool(active)
        self._zero_object_velocity()
        self._mujoco.mj_forward(self.model, self.data)

    def _set_target_visibility(self) -> None:
        if self.target_site_id < 0:
            return
        alpha = 0.0 if (self.render_mode == "rgb_array" and not self.target_visible_in_rgb_array) else 0.8
        self.model.site_rgba[self.target_site_id, 3] = alpha

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
            "grasp_relative_position": self._grasp_relative_position.copy(),
            "grasp_relative_quat": self._grasp_relative_quat.copy(),
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
        self._grasp_relative_position = np.asarray(state["grasp_relative_position"], dtype=np.float64).copy()
        self._grasp_relative_quat = np.asarray(state["grasp_relative_quat"], dtype=np.float64).copy()
        self._mujoco.mj_forward(self.model, self.data)

    def _apply_pre_action_plane_clip(self) -> None:
        if not self.constrain_robot_to_window_plane:
            return
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

    def _max_robot_plane_penetration(self) -> float:
        if not self.plane_robot_geom_ids:
            return 0.0
        plane_offset = float(np.dot(self.window_center, self.window_normal))
        max_penetration = float("-inf")
        for geom_id in self.plane_robot_geom_ids:
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

    def _build_collision_geom_sets(self) -> None:
        self.plane_robot_geom_ids = []
        self.plane_object_geom_ids = []
        self.plane_eef_geom_ids = []
        self.window_geom_ids = []
        self.object_collision_geom_ids = []
        self.eef_collision_geom_ids = []
        for geom_id in range(int(self.model.ngeom)):
            body_id = int(self.model.geom_bodyid[geom_id])
            if not self._is_collision_geom(geom_id):
                continue
            if body_id in self.robot_body_ids:
                self.plane_robot_geom_ids.append(geom_id)
                if body_id in self.hand_body_ids:
                    self.plane_eef_geom_ids.append(geom_id)
                    self.eef_collision_geom_ids.append(geom_id)
            elif body_id == self.obj_body_id:
                self.plane_object_geom_ids.append(geom_id)
                self.object_collision_geom_ids.append(geom_id)

            if body_id == self.window_body_id:
                self.window_geom_ids.append(geom_id)

    def _is_collision_geom(self, geom_id: int) -> bool:
        contype = int(self.model.geom_contype[geom_id])
        conaffinity = int(self.model.geom_conaffinity[geom_id])
        group = int(self.model.geom_group[geom_id])
        return bool(contype or conaffinity or group == 3)

    def _has_window_collision(self) -> bool:
        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self._is_window_collision_pair(geom1, geom2):
                return True
        return False

    def _is_window_collision_pair(self, geom1: int, geom2: int) -> bool:
        if geom1 in self.window_geom_ids:
            return geom2 in self.object_collision_geom_ids or geom2 in self.eef_collision_geom_ids
        if geom2 in self.window_geom_ids:
            return geom1 in self.object_collision_geom_ids or geom1 in self.eef_collision_geom_ids
        return False

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

    def _quat_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = self._normalize_vector(np.asarray(axis, dtype=np.float64))
        quat = np.empty(4, dtype=np.float64)
        self._mujoco.mju_axisAngle2Quat(quat, axis, float(angle))
        return quat

    def _quat_from_axis_alignment(
        self,
        current_axis: np.ndarray,
        target_axis: np.ndarray,
    ) -> np.ndarray:
        current_axis = self._normalize_vector(np.asarray(current_axis, dtype=np.float64))
        target_axis = self._normalize_vector(np.asarray(target_axis, dtype=np.float64))
        dot = float(np.clip(np.dot(current_axis, target_axis), -1.0, 1.0))
        cross = np.cross(current_axis, target_axis)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-8:
            if dot > 0.999:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(fallback, current_axis))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = self._normalize_vector(np.cross(current_axis, fallback))
            return self._quat_from_axis_angle(axis, np.pi)
        axis = cross / cross_norm
        angle = float(np.arctan2(cross_norm, dot))
        return self._quat_from_axis_angle(axis, angle)

    def _rotation_action_from_axes(
        self,
        current_axis: np.ndarray,
        target_axis: np.ndarray,
    ) -> np.ndarray:
        current_axis = self._normalize_vector(np.asarray(current_axis, dtype=np.float64))
        target_axis = self._normalize_vector(np.asarray(target_axis, dtype=np.float64))
        dot = float(np.clip(np.dot(current_axis, target_axis), -1.0, 1.0))
        cross = np.cross(current_axis, target_axis)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm < 1e-8:
            if dot > 0.999:
                return np.zeros(3, dtype=np.float64)
            fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(fallback, current_axis))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = self._normalize_vector(np.cross(current_axis, fallback))
            rotvec = axis * np.pi
        else:
            axis = cross / cross_norm
            angle = float(np.arctan2(cross_norm, dot))
            rotvec = axis * angle
        return np.clip(rotvec / max(self.rotation_action_scale, 1e-6), -1.0, 1.0)

    def _sample_random_axis(self) -> np.ndarray:
        axis = self.np_random.normal(size=3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return axis / norm

    def _quat_to_mat(self, quat: np.ndarray) -> np.ndarray:
        quat = self._normalize_vector(np.asarray(quat, dtype=np.float64))
        mat = np.empty(9, dtype=np.float64)
        self._mujoco.mju_quat2Mat(mat, quat)
        return mat.reshape(3, 3)

    def _quat_for_glass_normal(self, target_glass_normal: np.ndarray) -> np.ndarray:
        target_glass_normal = self._normalize_vector(np.asarray(target_glass_normal, dtype=np.float64))
        target_forward_axis = -target_glass_normal
        current_forward_axis = self.get_ee_forward_axis().copy()
        delta_quat = self._quat_from_axis_alignment(current_forward_axis, target_forward_axis)
        return self._normalize_vector(self._quat_multiply(delta_quat, self.get_mocap_quaternion()))

    def _zero_object_velocity(self) -> None:
        self.data.qvel[self.obj_dof_adr:self.obj_dof_adr + 6] = 0.0
