"""Task-independent Panda mocap control on MuJoCo."""

from __future__ import annotations

from typing import Any, Optional

import mujoco
import numpy as np
from gymnasium import Env, spaces
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.76,
    "azimuth": -1.0,
    "elevation": -25.5,
    "lookat": np.array([0.18, -0.04, 0.185]),
}


class FrankaEnv(MujocoRobotEnv):
    """Low-level Panda joint state and Cartesian mocap controller."""

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        *,
        model_path: str,
        n_substeps: int = 25,
        orientation_action_size: int = 3,
        position_action_scale: float = 0.05,
        rotation_action_scale: float = 0.20,
        policy_action_space: spaces.Space | None = None,
        policy_observation_space: spaces.Space | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.orientation_action_size = max(int(orientation_action_size), 0)
        self.position_action_scale = float(position_action_scale)
        self.rotation_action_scale = float(rotation_action_scale)
        self.neutral_joint_values = np.array(
            [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79],
            dtype=np.float64,
        )
        self.has_gripper_joints = False

        super().__init__(
            n_actions=3 + self.orientation_action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # MujocoRobotEnv internally constructs GoalEnv-style spaces from
        # ``_get_obs``. Keep those private for its initialization contract and
        # publish task-specific spaces once, before the concrete env returns.
        self._robot_action_space = self.action_space
        self._robot_observation_space = self.observation_space
        if policy_action_space is not None:
            self.action_space = policy_action_space
        if policy_observation_space is not None:
            self.observation_space = policy_observation_space
        self.ctrl_range = self.model.actuator_ctrlrange

    def _reset_robot_simulation(
        self,
        *,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Reset the RobotEnv internals without exposing its GoalEnv contract.

        ``GoalEnv.reset`` requires a public Dict space. Concrete flat-observation
        tasks instead keep that Dict private and call this equivalent reset path,
        so their public Box remains stable throughout reset.
        """

        Env.reset(self, seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        robot_observation = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return robot_observation

    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        free_joint_index = self._model_names.joint_names.index("obj_joint")
        robot_joint_names = self._model_names.joint_names[:free_joint_index]
        self.arm_joint_names = robot_joint_names[:7]
        self.gripper_joint_names = robot_joint_names[7:9]
        self.has_gripper_joints = len(self.gripper_joint_names) >= 2

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = self.data.qvel.copy()

    def _env_setup(self, neutral_joint_values: np.ndarray) -> None:
        self.set_joint_neutral()
        self.data.ctrl[:] = 0.0
        self.data.ctrl[:7] = neutral_joint_values[:7]
        self.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        self._mujoco_step()
        self.initial_object_height = float(
            self._utils.get_joint_qpos(
                self.model, self.data, "obj_joint"
            )[2]
        )

    def _set_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).copy()
        expected_shape = (3 + self.orientation_action_size,)
        if action.shape != expected_shape:
            raise ValueError(
                f"Expected low-level action shape {expected_shape}, got {action.shape}"
            )
        target_position = (
            self.get_ee_position()
            + action[:3] * self.position_action_scale
        )
        target_position[2] = max(float(target_position[2]), 0.0)
        target_quaternion = self.get_mocap_quaternion()
        if self.orientation_action_size:
            target_quaternion = self._apply_rotation_action(
                target_quaternion, action[3:]
            )
        self.set_mocap_pose(target_position, target_quaternion)

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        del action
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for index in range(model.eq_data.shape[0]):
                if model.eq_type[index] == mujoco.mjtEq.mjEQ_WELD:
                    model.eq_data[index, 3:10] = np.array(
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    )
        self._mujoco.mj_forward(model, data)

    def set_mocap_pose(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> None:
        self._utils.set_mocap_pos(
            self.model, self.data, "panda_mocap", position
        )
        self._utils.set_mocap_quat(
            self.model, self.data, "panda_mocap", orientation
        )

    def set_joint_neutral(self) -> None:
        for name, value in zip(
            self.arm_joint_names, self.neutral_joint_values
        ):
            self._utils.set_joint_qpos(
                self.model, self.data, name, value
            )
        if self.has_gripper_joints:
            for name in self.gripper_joint_names:
                self._utils.set_joint_qpos(
                    self.model, self.data, name, 0.04
                )

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()

    def get_ee_orientation(self) -> np.ndarray:
        site_matrix = self._utils.get_site_xmat(
            self.model, self.data, "ee_center_site"
        ).reshape(9, 1)
        quaternion = np.empty(4, dtype=np.float64)
        self._mujoco.mju_mat2Quat(quaternion, site_matrix)
        return quaternion

    def get_ee_rotation_matrix(self) -> np.ndarray:
        return self._utils.get_site_xmat(
            self.model, self.data, "ee_center_site"
        ).reshape(3, 3).copy()

    def get_ee_forward_axis(self) -> np.ndarray:
        return self._normalize_vector(-self.get_ee_rotation_matrix()[:, 2])

    def get_mocap_quaternion(self) -> np.ndarray:
        if self.data.mocap_quat.size == 0:
            return self.grasp_site_pose.copy()
        return np.asarray(self.data.mocap_quat[0], dtype=np.float64).copy()

    def get_object_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(
            self.model, self.data, "obj_site"
        ).copy()

    def get_object_rotation_matrix(self) -> np.ndarray:
        return self._utils.get_site_xmat(
            self.model, self.data, "obj_site"
        ).reshape(3, 3).copy()

    def get_body_state(self, name: str) -> np.ndarray:
        body_id = self._model_names.body_name2id[name]
        return np.concatenate(
            [self.data.xpos[body_id], self.data.xquat[body_id]]
        )

    def _joint_limit_violated(self) -> bool:
        """Return whether any limited MuJoCo joint is outside its range."""

        for joint_id in range(int(self.model.njnt)):
            if not bool(self.model.jnt_limited[joint_id]):
                continue
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            value = float(self.data.qpos[qpos_address])
            lower, upper = np.asarray(
                self.model.jnt_range[joint_id], dtype=np.float64
            )
            if value < lower - 1e-5 or value > upper + 1e-5:
                return True
        return False

    @staticmethod
    def _normalize_vector(
        vector: np.ndarray, eps: float = 1e-8
    ) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < eps:
            return np.zeros_like(vector)
        return vector / norm

    def _apply_rotation_action(
        self, current_quaternion: np.ndarray, rotation_action: np.ndarray
    ) -> np.ndarray:
        rotation_vector = (
            np.asarray(rotation_action, dtype=np.float64)
            * self.rotation_action_scale
        )
        angle = float(np.linalg.norm(rotation_vector))
        current_quaternion = self._normalize_vector(current_quaternion)
        if angle < 1e-8:
            return current_quaternion
        delta_quaternion = np.empty(4, dtype=np.float64)
        result = np.empty(4, dtype=np.float64)
        self._mujoco.mju_axisAngle2Quat(
            delta_quaternion, rotation_vector / angle, angle
        )
        self._mujoco.mju_mulQuat(
            result, delta_quaternion, current_quaternion
        )
        return self._normalize_vector(result)
