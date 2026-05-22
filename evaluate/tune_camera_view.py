from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import glfw
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import WindowViewer

import panda_mujoco_gym  # noqa: F401
from panda_mujoco_gym.envs.panda_env import DEFAULT_CAMERA_CONFIG


HELP_TEXT = """
Interactive camera tuner

Mouse:
  Left drag: rotate
  Shift + left drag: rotate horizontally
  Right drag: pan
  Mouse wheel: zoom

Keyboard:
  P: print the current camera config as DEFAULT_CAMERA_CONFIG
  R: reset camera to the repo default
  N: reset the scene and sample a new window pose
  H: print this help again

  , / . : azimuth - / +
  [ / ] : elevation - / +
  - / = : zoom out / in

  U / O: lookat x - / +
  J / L: lookat y - / +
  I / K: lookat z + / -

Close the viewer window, press Esc, or press Ctrl+C in the terminal to exit.
""".strip()


@dataclass
class CameraState:
    distance: float
    azimuth: float
    elevation: float
    lookat: np.ndarray


def default_camera_state() -> CameraState:
    return CameraState(
        distance=float(DEFAULT_CAMERA_CONFIG["distance"]),
        azimuth=float(DEFAULT_CAMERA_CONFIG["azimuth"]),
        elevation=float(DEFAULT_CAMERA_CONFIG["elevation"]),
        lookat=np.asarray(DEFAULT_CAMERA_CONFIG["lookat"], dtype=np.float64).copy(),
    )


class CameraTunerViewer(WindowViewer):
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, tuner: "InteractiveCameraTuner") -> None:
        self._tuner = tuner
        super().__init__(model, data)

    def _key_callback(self, window, key: int, scancode, action: int, mods) -> None:
        if action != glfw.RELEASE:
            return

        if key == glfw.KEY_P:
            self._tuner.print_camera_state(self.cam)
            return
        if key == glfw.KEY_H:
            print(f"\n{HELP_TEXT}")
            return
        if key == glfw.KEY_N:
            self._tuner.reset_scene()
            self._tuner.print_scene_summary()
            return
        if key == glfw.KEY_R:
            self._tuner.reset_camera()
            self._tuner.print_camera_state(self.cam)
            return
        if key == glfw.KEY_COMMA:
            self._tuner.adjust_camera(azimuth_delta=-self._tuner.angle_step_deg)
            return
        if key == glfw.KEY_PERIOD:
            self._tuner.adjust_camera(azimuth_delta=self._tuner.angle_step_deg)
            return
        if key == glfw.KEY_LEFT_BRACKET:
            self._tuner.adjust_camera(elevation_delta=-self._tuner.angle_step_deg)
            return
        if key == glfw.KEY_RIGHT_BRACKET:
            self._tuner.adjust_camera(elevation_delta=self._tuner.angle_step_deg)
            return
        if key == glfw.KEY_MINUS:
            self._tuner.adjust_camera(distance_delta=self._tuner.distance_step)
            return
        if key == glfw.KEY_EQUAL:
            self._tuner.adjust_camera(distance_delta=-self._tuner.distance_step)
            return
        if key == glfw.KEY_U:
            self._tuner.adjust_camera(lookat_delta=np.array([-self._tuner.lookat_step, 0.0, 0.0], dtype=np.float64))
            return
        if key == glfw.KEY_O:
            self._tuner.adjust_camera(lookat_delta=np.array([self._tuner.lookat_step, 0.0, 0.0], dtype=np.float64))
            return
        if key == glfw.KEY_J:
            self._tuner.adjust_camera(lookat_delta=np.array([0.0, -self._tuner.lookat_step, 0.0], dtype=np.float64))
            return
        if key == glfw.KEY_L:
            self._tuner.adjust_camera(lookat_delta=np.array([0.0, self._tuner.lookat_step, 0.0], dtype=np.float64))
            return
        if key == glfw.KEY_I:
            self._tuner.adjust_camera(lookat_delta=np.array([0.0, 0.0, self._tuner.lookat_step], dtype=np.float64))
            return
        if key == glfw.KEY_K:
            self._tuner.adjust_camera(lookat_delta=np.array([0.0, 0.0, -self._tuner.lookat_step], dtype=np.float64))
            return

        super()._key_callback(window, key, scancode, action, mods)

    def _create_overlay(self) -> None:
        super()._create_overlay()
        top_right = getattr(mujoco.mjtGridPos, "mjGRID_TOPRIGHT", mujoco.mjtGridPos.mjGRID_TOPLEFT)
        bottom_right = getattr(mujoco.mjtGridPos, "mjGRID_BOTTOMRIGHT", mujoco.mjtGridPos.mjGRID_BOTTOMLEFT)

        self.add_overlay(top_right, "Camera tuner", "[P] print  [R] reset  [N] new scene  [H] help")
        self.add_overlay(top_right, "Azimuth / Elevation", "[,][.] / [[] []]")
        self.add_overlay(top_right, "Zoom", "[-] out  [=] in")
        self.add_overlay(top_right, "Lookat x / y / z", "[U][O] / [J][L] / [I][K]")
        self.add_overlay(
            bottom_right,
            "Camera",
            f"d={float(self.cam.distance):.3f} az={float(self.cam.azimuth):.1f} el={float(self.cam.elevation):.1f}",
        )
        self.add_overlay(
            bottom_right,
            "Lookat",
            f"[{float(self.cam.lookat[0]):.3f}, {float(self.cam.lookat[1]):.3f}, {float(self.cam.lookat[2]):.3f}]",
        )


class InteractiveCameraTuner:
    def __init__(self, env_id: str, seed: Optional[int]) -> None:
        self.env = gym.make(env_id, render_mode=None, target_visible_in_rgb_array=True)
        self.base = self.env.unwrapped
        self.default_camera = default_camera_state()
        self.viewer: Optional[CameraTunerViewer] = None
        self.next_seed = seed

        self.angle_step_deg = 2.0
        self.distance_step = 0.05
        self.lookat_step = 0.02

        self.reset_scene()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.env.close()

    def reset_scene(self) -> None:
        if self.next_seed is None:
            self.env.reset()
        else:
            self.env.reset(seed=self.next_seed)
            self.next_seed += 1
        self.base = self.env.unwrapped
        mujoco.mj_forward(self.base.model, self.base.data)
        if self.viewer is not None:
            self.viewer.model = self.base.model
            self.viewer.data = self.base.data

    def apply_camera_state(self, cam: mujoco.MjvCamera, state: CameraState) -> None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.distance = float(max(0.05, state.distance))
        cam.azimuth = float(state.azimuth)
        cam.elevation = float(state.elevation)
        cam.lookat[:] = np.asarray(state.lookat, dtype=np.float64)

    def capture_camera_state(self, cam: mujoco.MjvCamera) -> CameraState:
        return CameraState(
            distance=float(cam.distance),
            azimuth=float(cam.azimuth),
            elevation=float(cam.elevation),
            lookat=np.asarray(cam.lookat, dtype=np.float64).copy(),
        )

    def adjust_camera(
        self,
        *,
        azimuth_delta: float = 0.0,
        elevation_delta: float = 0.0,
        distance_delta: float = 0.0,
        lookat_delta: Optional[np.ndarray] = None,
    ) -> None:
        if self.viewer is None:
            return
        cam = self.viewer.cam
        cam.azimuth += float(azimuth_delta)
        cam.elevation += float(elevation_delta)
        cam.distance = float(max(0.05, cam.distance + distance_delta))
        if lookat_delta is not None:
            cam.lookat[:] = np.asarray(cam.lookat, dtype=np.float64) + np.asarray(lookat_delta, dtype=np.float64)

    def reset_camera(self) -> None:
        if self.viewer is None:
            return
        self.apply_camera_state(self.viewer.cam, self.default_camera)

    def print_scene_summary(self) -> None:
        print("\nScene state:")
        if hasattr(self.base, "get_window_center"):
            print(f"  window_center: {self._format_vec(self.base.get_window_center())}")
        if hasattr(self.base, "pickup_station_center"):
            print(f"  pickup_station_center: {self._format_vec(self.base.pickup_station_center)}")
        if hasattr(self.base, "get_object_position"):
            print(f"  object_position: {self._format_vec(self.base.get_object_position())}")

    def print_camera_state(self, cam: mujoco.MjvCamera) -> None:
        state = self.capture_camera_state(cam)
        print("\nCurrent free camera:")
        print(f"  distance:  {state.distance:.6f}")
        print(f"  azimuth:   {state.azimuth:.6f}")
        print(f"  elevation: {state.elevation:.6f}")
        print(f"  lookat:    {self._format_vec(state.lookat)}")
        print("\nPaste this into panda_mujoco_gym/envs/panda_env.py:")
        print("DEFAULT_CAMERA_CONFIG = {")
        print(f'    "distance": {state.distance:.6f},')
        print(f'    "azimuth": {state.azimuth:.6f},')
        print(f'    "elevation": {state.elevation:.6f},')
        print(
            '    "lookat": np.array(['
            f"{state.lookat[0]:.6f}, {state.lookat[1]:.6f}, {state.lookat[2]:.6f}"
            "], dtype=np.float64),"
        )
        print("}")

    def run(self) -> None:
        print(HELP_TEXT)
        print("\nThe viewer starts from panda_mujoco_gym/envs/panda_env.py::DEFAULT_CAMERA_CONFIG.")
        self.print_scene_summary()

        self.viewer = CameraTunerViewer(self.base.model, self.base.data, self)
        self.reset_camera()
        self.print_camera_state(self.viewer.cam)

        while self.viewer.window is not None and not glfw.window_should_close(self.viewer.window):
            self.viewer.render()
            time.sleep(1.0 / 60.0)

    @staticmethod
    def _format_vec(vec: np.ndarray) -> str:
        values = np.asarray(vec, dtype=np.float64).reshape(-1)
        return "[" + ", ".join(f"{value:.6f}" for value in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo camera tuner for the window task.")
    parser.add_argument(
        "--env",
        default="FrankaPickAndPlaceWindowSparse-v0",
        help="Gymnasium env id to load.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Initial reset seed. Use a different value to inspect another sampled window pose.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuner = InteractiveCameraTuner(env_id=args.env, seed=args.seed)
    try:
        tuner.run()
    finally:
        tuner.close()


if __name__ == "__main__":
    main()
