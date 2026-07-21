#/home/minjun/panda_mujoco_gym/panda_mujoco_gym/envs/panda_env.py
import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat

# DEFAULT_CAMERA_CONFIG = {
#     "distance": 2.76,
#     "azimuth": -1.0,
#     "elevation": -25.5,
#     "lookat": np.array([0.18, -0.04, 0.185]),
# }

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.880000,
    "azimuth": -226.000000,
    "elevation": -18.000000,
    "lookat": np.array([-0.000000, -0.130000, 0.230000], dtype=np.float64),
}

class FrankaEnv(MujocoRobotEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: str = None,
        n_substeps: int = 50,
        reward_type: str = "sparse",
        block_gripper: bool = False,
        include_gripper_action: bool = True,
        fixed_gripper_action: float = 0.0,
        terminate_on_success: bool = False,
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        obj_xy_range: float = 0.3,
        goal_x_offset: float = 0.4,
        goal_z_range: float = 0.2,
        orientation_action_size: int = 0,
        position_action_scale: float = 0.05,
        rotation_action_scale: float = 0.20,
        **kwargs,
    ):
        self.block_gripper = block_gripper
        self.include_gripper_action = bool(include_gripper_action) and not self.block_gripper
        self.fixed_gripper_action = float(fixed_gripper_action)
        self.terminate_on_success = bool(terminate_on_success)
        self.model_path = model_path
        self.orientation_action_size = int(max(orientation_action_size, 0))
        self.control_orientation = self.orientation_action_size > 0
        self.position_action_scale = float(position_action_scale)
        self.rotation_action_scale = float(rotation_action_scale)

        action_size = 3
        action_size += self.orientation_action_size
        action_size += 1 if self.include_gripper_action else 0

        self.reward_type = reward_type
        self.has_gripper_joints = False

        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79], dtype=np.float64)

        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.distance_threshold = distance_threshold

        # sample areas for the object and goal target
        self.obj_xy_range = obj_xy_range
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        self.goal_range_low = np.array([-self.goal_xy_range / 2 + goal_x_offset, -self.goal_xy_range / 2, 0])
        self.goal_range_high = np.array([self.goal_xy_range / 2 + goal_x_offset, self.goal_xy_range / 2, self.goal_z_range])
        self.obj_range_low = np.array([-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0])
        self.obj_range_high = np.array([self.obj_xy_range / 2, self.obj_xy_range / 2, 0])

        self.goal_range_low[0] += 0.6
        self.goal_range_high[0] += 0.6
        self.obj_range_low[0] += 0.6
        self.obj_range_high[0] += 0.6

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        # control range
        self.ctrl_range = self.model.actuator_ctrlrange

    # override the methods in MujocoRobotEnv
    # -----------------------------
    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # index used to distinguish arm and gripper joints
        free_joint_index = self._model_names.joint_names.index("obj_joint")
        self.arm_joint_names = self._model_names.joint_names[:free_joint_index][0:7]
        self.gripper_joint_names = self._model_names.joint_names[:free_joint_index][7:9]
        self.has_gripper_joints = len(self.gripper_joint_names) >= 2

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        self.set_joint_neutral()
        self.data.ctrl[:] = 0.0
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)

        self._mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco_step()

        self.initial_object_height = self._utils.get_joint_qpos(self.model, self.data, "obj_joint")[2].copy()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        ee_position = obs["observation"][:3]
        achieved_goal = obs["achieved_goal"]

        info = {
            "is_success": self._is_success(achieved_goal, self.goal),
            "ee_object_distance": float(np.linalg.norm(ee_position - achieved_goal)),
            "object_height": float(achieved_goal[2] - self.initial_object_height),
        }
        terminated = bool(info["is_success"]) if self.terminate_on_success else False
        truncated = bool(self.compute_truncated(achieved_goal, self.goal, info))
        reward = self.compute_reward(achieved_goal, self.goal, info)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)

        reward = -d

        # 对于 pick-and-place 类任务，仅靠目标距离几乎无法探索，
        # 因此额外加入较弱的 reach 和 lift 信号。
        if not self.block_gripper:
            ee_object_distance = float(info.get("ee_object_distance", 0.0))
            object_height = max(float(info.get("object_height", 0.0)), 0.0)

            reward -= 0.25 * ee_object_distance

            if self.goal_z_range > 0.0:
                lift_cap = max(self.goal_z_range, self.distance_threshold)
                reward += 0.5 * min(object_height, lift_cap)
                if object_height > self.distance_threshold:
                    reward += 0.25

        return reward

    def _set_action(self, action) -> None:
        action = action.copy()
        pos_ctrl = action[:3]
        rot_ctrl = None
        if self.control_orientation:
            rot_start = 3
            rot_end = rot_start + self.orientation_action_size
            rot_ctrl = action[rot_start:rot_end]
        if self.include_gripper_action:
            suction_ctrl = float(action[3 + self.orientation_action_size])
        else:
            suction_ctrl = self.fixed_gripper_action

        if self.has_gripper_joints:
            open_fraction = np.clip((suction_ctrl + 1.0) * 0.5, 0.0, 1.0)
            target_width = open_fraction * 0.08
            half_width = np.clip(target_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])
            self.data.ctrl[-2:] = half_width

        # control the end-effector with mocap body
        pos_ctrl *= self.position_action_scale
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = np.max((0, pos_ctrl[2]))

        target_quat = self.grasp_site_pose.copy()
        if self.control_orientation and rot_ctrl is not None:
            target_quat = self._apply_rotation_action(self.get_mocap_quaternion(), rot_ctrl)

        self.set_mocap_pose(pos_ctrl, target_quat)

    def _get_obs(self) -> dict:
        # robot
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()

        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        if not self.block_gripper:
            suction_state = np.array([self.get_gripper_state()], dtype=np.float64)

        # object
        # object cartesian position: 3
        object_position = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()

        # object rotations: 3
        object_rotation = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "obj_site")).copy()

        # object linear velocities
        object_velp = self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy() * self.dt

        # object angular velocities
        object_velr = self._utils.get_site_xvelr(self.model, self.data, "obj_site").copy() * self.dt

        if not self.block_gripper:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        suction_state,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": self.goal.copy(),
            }
        else:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": self.goal.copy(),
            }

        return obs

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self) -> None:
        # visualize goal site
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._sample_object()

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # custom methods
    # -----------------------------
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    # relative pose
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._mujoco.mj_forward(model, data)

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def get_object_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()

    def minimum_goal_object_distance(self) -> float:
        if self.reward_type != "sparse":
            return 0.0
        return float(self.distance_threshold + 0.02)

    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

        if self.has_gripper_joints:
            for name in self.gripper_joint_names:
                self._utils.set_joint_qpos(self.model, self.data, name, 0.04)

    def _sample_goal(self) -> np.ndarray:
        object_position = self.get_object_position()
        min_distance = self.minimum_goal_object_distance()
        goal_dtype = object_position.dtype

        best_goal = None
        best_distance = float("-inf")

        for _ in range(64):
            goal = np.array([0.0, 0.0, self.initial_object_height], dtype=np.float64)
            noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
            if not self.block_gripper and self.goal_z_range > 0.0:
                if self.np_random.random() < 0.7:
                    noise[2] = 0.0
            goal += noise

            distance = float(np.linalg.norm(goal - object_position))
            if distance > best_distance:
                best_goal = goal.copy()
                best_distance = distance

            if distance >= min_distance:
                return goal.astype(goal_dtype)

        return np.asarray(best_goal, dtype=goal_dtype)

    def _sample_object(self) -> None:
        object_position = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat

    def get_ee_rotation_matrix(self) -> np.ndarray:
        return self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(3, 3).copy()

    def get_ee_forward_axis(self) -> np.ndarray:
        # The suction tool's approach normal points opposite to the local +Z
        # axis of ee_center_site in the current XML setup.
        return self._normalize_vector(-self.get_ee_rotation_matrix()[:, 2])

    def get_object_rotation_matrix(self) -> np.ndarray:
        return self._utils.get_site_xmat(self.model, self.data, "obj_site").reshape(3, 3).copy()

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_mocap_quaternion(self) -> np.ndarray:
        if getattr(self.data, "mocap_quat", None) is None or self.data.mocap_quat.size == 0:
            return self.grasp_site_pose.copy()
        return np.asarray(self.data.mocap_quat[0], dtype=np.float64).copy()

    def get_body_state(self, name) -> np.ndarray:
        body_id = self._model_names.body_name2id[name]
        body_xpos = self.data.xpos[body_id]
        body_xquat = self.data.xquat[body_id]
        body_state = np.concatenate([body_xpos, body_xquat])
        return body_state

    def get_gripper_state(self) -> float:
        if not self.has_gripper_joints:
            return 1.0
        finger1 = float(self._utils.get_joint_qpos(self.model, self.data, self.gripper_joint_names[0]))
        finger2 = float(self._utils.get_joint_qpos(self.model, self.data, self.gripper_joint_names[1]))
        return finger1 + finger2

    def _normalize_vector(self, vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < eps:
            return np.zeros_like(vector)
        return vector / norm

    def _apply_rotation_action(self, current_quat: np.ndarray, rot_ctrl: np.ndarray) -> np.ndarray:
        rotvec = np.asarray(rot_ctrl, dtype=np.float64) * self.rotation_action_scale
        angle = float(np.linalg.norm(rotvec))
        current_quat = self._normalize_vector(current_quat)
        if angle < 1e-8:
            return current_quat

        axis = rotvec / angle
        delta_quat = np.empty(4, dtype=np.float64)
        composed_quat = np.empty(4, dtype=np.float64)
        self._mujoco.mju_axisAngle2Quat(delta_quat, axis, angle)
        self._mujoco.mju_mulQuat(composed_quat, delta_quat, current_quat)
        return self._normalize_vector(composed_quat)
