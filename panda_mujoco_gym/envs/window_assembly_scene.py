"""Physical MuJoCo scene services for subway window-glass assembly.

This module owns scene setup, suction attachment, collision queries, and
simulation state management. It deliberately contains no RL task, reward,
reset mode, or policy observation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from panda_mujoco_gym.envs.panda_env import FrankaEnv


MODEL_XML_PATH = str(
    Path(__file__).resolve().parent.parent / "assets" / "window_assembly.xml"
)


class WindowAssemblyScene(FrankaEnv):
    """Panda, window frame, glass, and suction-cup simulation services."""

    def __init__(
        self,
        *,
        reward_type: str = "dense",
        target_visible_in_rgb_array: bool = True,
        **kwargs: Any,
    ) -> None:
        self.target_visible_in_rgb_array = bool(target_visible_in_rgb_array)

        self.window_goal_height = 0.65
        self.window_ring_radius = 0.80
        self.window_sector_min_angle_deg = -60.0
        self.window_sector_max_angle_deg = 0.0
        self.window_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.window_opening_size = np.array([0.36, 0.54], dtype=np.float64)
        self.glass_thickness = 0.004
        self.glass_size_xy = self.window_opening_size - 0.01
        self.glass_half_extents = np.array(
            [
                self.glass_size_xy[0] / 2.0,
                self.glass_size_xy[1] / 2.0,
                self.glass_thickness / 2.0,
            ],
            dtype=np.float64,
        )
        self.target_glass_normal = self.window_normal.copy()

        self.pickup_station_center = np.array(
            [0.0, -0.60, 0.25], dtype=np.float64
        )
        self.object_settle_steps = 120
        self.object_settle_linvel_threshold = 5e-3
        self.object_settle_angvel_threshold = 5e-2
        self.post_grasp_canonical_goal_margin = 0.14
        self.post_grasp_min_lift_above_object = 0.18
        self.post_grasp_motion_steps = 12
        self.plane_constraint_eps = 1e-4

        self.is_attached = False
        self._grasp_relative_position = np.zeros(3, dtype=np.float64)
        self._grasp_relative_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.window_body_id = -1
        self.obj_body_id = -1
        self.ee_center_body_id = -1
        self.obj_joint_id = -1
        self.obj_qpos_adr = -1
        self.obj_dof_adr = -1
        self.grasp_weld_eq_id = -1
        self.link0_body_id = -1
        self.hand_body_id = -1
        self.target_site_id = -1
        self.robot_body_ids: set[int] = set()
        self.hand_body_ids: set[int] = set()
        self.plane_object_geom_ids: list[int] = []
        self.window_geom_ids: list[int] = []
        self.object_collision_geom_ids: list[int] = []
        self.eef_collision_geom_ids: list[int] = []
        self.mesh_vertices_by_mesh_id: dict[int, np.ndarray] = {}
        self.window_center = np.array(
            [self.window_ring_radius, 0.0, self.window_goal_height],
            dtype=np.float64,
        )
        self.window_safe_goal = self.window_center.copy()
        self._last_plane_violation = False
        self._last_max_plane_penetration = 0.0
        self._last_collision = False
        self._last_collision_with_window = False

        super().__init__(
            model_path=MODEL_XML_PATH,
            n_substeps=25,
            orientation_action_size=3,
            position_action_scale=0.05,
            rotation_action_scale=0.20,
            **kwargs,
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
            self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_EQUALITY,
                "obj_grasp_weld",
            )
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
        self.window_safe_goal = self.window_center.copy()
        self._set_target_visibility()
        self._mujoco.mj_forward(self.model, self.data)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Internal GoalEnv state; the fine-align env replaces it publicly."""

        ee_position = self.get_ee_position().copy()
        ee_velocity = (
            self._utils.get_site_xvelp(
                self.model, self.data, "ee_center_site"
            ).copy()
            * self.dt
        )
        object_position = self.get_object_position().copy()
        glass_normal = self.get_glass_normal()
        desired_goal = (
            self.goal.copy()
            if self.goal.shape == (6,)
            else self._compose_window_goal(self.window_center)
        )
        achieved_goal = np.concatenate(
            [object_position, glass_normal]
        ).astype(np.float64)
        observation = np.concatenate(
            [
                ee_position,
                ee_velocity,
                desired_goal[:3] - object_position,
                desired_goal[3:6],
                self.get_ee_forward_axis(),
                glass_normal,
                np.array([float(self.is_attached)], dtype=np.float64),
            ]
        ).astype(np.float64)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal.astype(np.float64),
        }

    def _sample_goal(self) -> np.ndarray:
        self.window_safe_goal = self.window_center.copy()
        return self._compose_window_goal(self.window_center)

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        del achieved_goal, desired_goal
        return np.float32(False)

    def compute_reward(self, achieved_goal, desired_goal, info) -> np.float32:
        del achieved_goal, desired_goal, info
        return np.float32(0.0)

    def _render_callback(self) -> None:
        self._set_target_visibility()

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
        return np.concatenate(
            [
                np.asarray(window_center, dtype=np.float64).copy(),
                self.target_glass_normal.copy(),
            ]
        ).astype(np.float64)

    def _sample_object(self) -> None:
        object_pose = np.concatenate(
            [
                self.pickup_station_center.copy(),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            ]
        )
        self._utils.set_joint_qpos(
            self.model, self.data, "obj_joint", object_pose
        )
        self._zero_object_velocity()

    def _settle_object(self) -> None:
        settled_steps = 0
        for _ in range(self.object_settle_steps):
            self._mujoco_step()
            linear_speed = float(
                np.linalg.norm(
                    self._utils.get_site_xvelp(
                        self.model, self.data, "obj_site"
                    )
                )
            )
            angular_speed = float(
                np.linalg.norm(
                    self._utils.get_site_xvelr(
                        self.model, self.data, "obj_site"
                    )
                )
            )
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

    def _stabilize_reset_state(self) -> None:
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def get_glass_normal(self) -> np.ndarray:
        return self._normalize_vector(
            self.get_object_rotation_matrix()[:, 2]
        )

    def get_window_center(self) -> np.ndarray:
        return np.asarray(self.window_center, dtype=np.float64).copy()

    def get_object_position(self) -> np.ndarray:
        if self.is_attached:
            ee_position = self.data.xpos[self.ee_center_body_id].copy()
            ee_rotation = self.data.xmat[self.ee_center_body_id].reshape(3, 3)
            return ee_position + ee_rotation @ self._grasp_relative_position
        return super().get_object_position()

    def get_object_rotation_matrix(self) -> np.ndarray:
        if self.is_attached:
            ee_rotation = self.data.xmat[self.ee_center_body_id].reshape(3, 3)
            relative_rotation = self._quat_to_matrix(
                self._grasp_relative_quat
            )
            return (ee_rotation @ relative_rotation).astype(np.float64)
        return super().get_object_rotation_matrix()

    def _activate_grasp_weld(self) -> None:
        ee_position = self.data.xpos[self.ee_center_body_id].copy()
        ee_quaternion = self.data.xquat[self.ee_center_body_id].copy()
        object_position = self.data.xpos[self.obj_body_id].copy()
        object_quaternion = self.data.xquat[self.obj_body_id].copy()
        ee_rotation = self.data.xmat[self.ee_center_body_id].reshape(3, 3)
        relative_position = ee_rotation.T @ (object_position - ee_position)
        relative_quaternion = self._quat_multiply(
            self._quat_conjugate(ee_quaternion), object_quaternion
        )
        self.model.eq_data[self.grasp_weld_eq_id, :3] = 0.0
        self.model.eq_data[self.grasp_weld_eq_id, 3:6] = relative_position
        self.model.eq_data[self.grasp_weld_eq_id, 6:10] = self._normalize_vector(
            relative_quaternion
        )
        self.model.eq_data[self.grasp_weld_eq_id, 10] = 1.0
        self._grasp_relative_position = relative_position.copy()
        self._grasp_relative_quat = self._normalize_vector(relative_quaternion)
        self._set_grasp_weld_active(True)

    def _set_grasp_weld_active(self, active: bool) -> None:
        if self.grasp_weld_eq_id < 0:
            return
        self.model.eq_active[self.grasp_weld_eq_id] = np.uint8(
            1 if active else 0
        )
        self.is_attached = bool(active)
        self._zero_object_velocity()
        self._mujoco.mj_forward(self.model, self.data)

    def _set_target_visibility(self) -> None:
        if self.target_site_id < 0:
            return
        render_mode = getattr(self, "render_mode", None)
        alpha = (
            0.0
            if render_mode == "rgb_array"
            and not self.target_visible_in_rgb_array
            else 0.8
        )
        self.model.site_rgba[self.target_site_id, 3] = alpha

    def _capture_sim_state(self) -> dict[str, Any]:
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

    def _restore_sim_state(self, state: dict[str, Any]) -> None:
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
        self._grasp_relative_position = np.asarray(
            state["grasp_relative_position"], dtype=np.float64
        ).copy()
        self._grasp_relative_quat = np.asarray(
            state["grasp_relative_quat"], dtype=np.float64
        ).copy()
        self._mujoco.mj_forward(self.model, self.data)

    def _max_object_penetration(self) -> float:
        if not self.plane_object_geom_ids:
            return 0.0
        plane_offset = float(np.dot(self.window_center, self.window_normal))
        max_penetration = max(
            self._geom_support_on_normal(geom_id) - plane_offset
            for geom_id in self.plane_object_geom_ids
        )
        return max(0.0, float(max_penetration))

    def _geom_support_on_normal(self, geom_id: int) -> float:
        normal = self.window_normal
        geom_position = np.asarray(self.data.geom_xpos[geom_id], dtype=np.float64)
        geom_rotation = np.asarray(
            self.data.geom_xmat[geom_id], dtype=np.float64
        ).reshape(3, 3)
        local_direction = geom_rotation.T @ normal
        geom_type = int(self.model.geom_type[geom_id])
        geom_size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        center_projection = float(np.dot(geom_position, normal))
        geom_enum = self._mujoco.mjtGeom
        if geom_type == int(geom_enum.mjGEOM_BOX):
            radius = float(np.dot(np.abs(local_direction[:3]), geom_size[:3]))
        elif geom_type == int(geom_enum.mjGEOM_SPHERE):
            radius = float(geom_size[0])
        elif geom_type in (
            int(geom_enum.mjGEOM_CAPSULE),
            int(geom_enum.mjGEOM_CYLINDER),
        ):
            radius = float(
                geom_size[0] * np.linalg.norm(local_direction[:2])
                + geom_size[1] * abs(local_direction[2])
            )
        elif geom_type == int(geom_enum.mjGEOM_ELLIPSOID):
            radius = float(np.linalg.norm(geom_size[:3] * local_direction[:3]))
        elif geom_type == int(geom_enum.mjGEOM_MESH):
            mesh_id = int(self.model.geom_dataid[geom_id])
            vertices = self.mesh_vertices_by_mesh_id.get(mesh_id)
            radius = (
                float(np.max(vertices @ local_direction))
                if vertices is not None and vertices.size
                else float(self.model.geom_rbound[geom_id])
            )
        else:
            radius = float(self.model.geom_rbound[geom_id])
        return center_projection + radius

    def _build_mesh_vertex_cache(self) -> dict[int, np.ndarray]:
        cache: dict[int, np.ndarray] = {}
        for mesh_id in range(int(self.model.nmesh)):
            address = int(self.model.mesh_vertadr[mesh_id])
            count = int(self.model.mesh_vertnum[mesh_id])
            cache[mesh_id] = np.asarray(
                self.model.mesh_vert[address : address + count],
                dtype=np.float64,
            ).copy()
        return cache

    def _build_collision_geom_sets(self) -> None:
        self.plane_object_geom_ids = []
        self.window_geom_ids = []
        self.object_collision_geom_ids = []
        self.eef_collision_geom_ids = []
        for geom_id in range(int(self.model.ngeom)):
            if not self._is_collision_geom(geom_id):
                continue
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id == self.obj_body_id:
                self.plane_object_geom_ids.append(geom_id)
                self.object_collision_geom_ids.append(geom_id)
            if body_id in self.hand_body_ids:
                self.eef_collision_geom_ids.append(geom_id)
            if body_id == self.window_body_id:
                self.window_geom_ids.append(geom_id)

    def _is_collision_geom(self, geom_id: int) -> bool:
        return bool(
            int(self.model.geom_contype[geom_id])
            or int(self.model.geom_conaffinity[geom_id])
            or int(self.model.geom_group[geom_id]) == 3
        )

    def _has_window_collision(self) -> bool:
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            if self._is_window_collision_pair(
                int(contact.geom1), int(contact.geom2)
            ):
                return True
        return False

    def _is_window_collision_pair(self, geom1: int, geom2: int) -> bool:
        collision_geoms = (
            self.object_collision_geom_ids + self.eef_collision_geom_ids
        )
        return bool(
            (geom1 in self.window_geom_ids and geom2 in collision_geoms)
            or (geom2 in self.window_geom_ids and geom1 in collision_geoms)
        )

    def _body_descendants(self, root_body_id: int) -> set[int]:
        descendants: set[int] = set()
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

    @staticmethod
    def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(quaternion, dtype=np.float64)
        return np.array(
            [
                quaternion[0],
                -quaternion[1],
                -quaternion[2],
                -quaternion[3],
            ],
            dtype=np.float64,
        )

    def _quat_multiply(
        self, first: np.ndarray, second: np.ndarray
    ) -> np.ndarray:
        result = np.empty(4, dtype=np.float64)
        self._mujoco.mju_mulQuat(result, first, second)
        return result

    def _quat_to_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        matrix = np.empty(9, dtype=np.float64)
        self._mujoco.mju_quat2Mat(
            matrix, self._normalize_vector(np.asarray(quaternion, dtype=np.float64))
        )
        return matrix.reshape(3, 3)

    def _zero_object_velocity(self) -> None:
        if self.obj_dof_adr >= 0:
            self.data.qvel[self.obj_dof_adr : self.obj_dof_adr + 6] = 0.0
