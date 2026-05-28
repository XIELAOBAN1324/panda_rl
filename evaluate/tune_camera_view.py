from __future__ import annotations

import argparse
import ctypes
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
  Hold Left Shift: move faster
  Hold Left Ctrl: move slower

  Hold , / . : azimuth - / +
  Hold [ / ] : elevation - / +
  Hold - / = : zoom out / in

  Hold U / O or Left / Right: lookat x - / +
  Hold J / L or Down / Up: lookat y - / +
  Hold I / K or PageUp / PageDown: lookat z + / -

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
        self._pressed_keys: set[int] = set()
        super().__init__(model, data)

    def render(self) -> None:
        self._tuner.update_keyboard_camera_controls()
        super().render()

    def _key_callback(self, window, key: int, scancode, action: int, mods) -> None:
        if action in (glfw.PRESS, glfw.REPEAT):
            self._pressed_keys.add(key)
        elif action == glfw.RELEASE:
            self._pressed_keys.discard(key)

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
        if key in self._tuner.continuous_camera_keys:
            return

        super()._key_callback(window, key, scancode, action, mods)

    def is_key_pressed(self, key: int) -> bool:
        if key in self._pressed_keys:
            return True
        if self.window is None:
            return False
        return glfw.get_key(self.window, key) in (glfw.PRESS, glfw.REPEAT)

    def _create_overlay(self) -> None:
        super()._create_overlay()
        top_right = getattr(mujoco.mjtGridPos, "mjGRID_TOPRIGHT", mujoco.mjtGridPos.mjGRID_TOPLEFT)
        bottom_right = getattr(mujoco.mjtGridPos, "mjGRID_BOTTOMRIGHT", mujoco.mjtGridPos.mjGRID_BOTTOMLEFT)

        self.add_overlay(top_right, "Camera tuner", "[P] print  [R] reset  [N] new scene  [H] help")
        self.add_overlay(top_right, "Azimuth / Elevation", "hold [,][.] / [[] []]")
        self.add_overlay(top_right, "Zoom", "hold [-] out  [=] in")
        self.add_overlay(top_right, "Lookat x / y / z", "hold [U][O] / [J][L] / [I][K]")
        self.add_overlay(top_right, "Lookat alt", "arrows for x/y  PgUp/PgDn for z")
        self.add_overlay(top_right, "Speed", "Shift fast  Ctrl slow")
        self.add_overlay(
            bottom_right,
            "Camera",
            f"d={float(self.cam.distance):.3f} az={float(self.cam.azimuth):.1f} el={float(self.cam.elevation):.1f}",
        )
        self.add_overlay(
            bottom_right,
            "Lookat",
            self._tuner.format_camera_lookat(self.cam, precision=3),
        )


class InteractiveCameraTuner:
    def __init__(self, env_id: str, seed: Optional[int]) -> None:
        self.env = gym.make(env_id, render_mode=None, target_visible_in_rgb_array=True)
        self.base = self.env.unwrapped
        self.default_camera = default_camera_state()
        self.viewer: Optional[CameraTunerViewer] = None
        self.next_seed = seed
        self._last_keyboard_update_time: Optional[float] = None

        self.angle_speed_deg = 75.0
        self.distance_speed = 1.2
        self.lookat_speed = 0.45
        self.fast_multiplier = 4.0
        self.slow_multiplier = 0.25
        self.continuous_camera_keys = {
            glfw.KEY_COMMA,
            glfw.KEY_PERIOD,
            glfw.KEY_LEFT_BRACKET,
            glfw.KEY_RIGHT_BRACKET,
            glfw.KEY_MINUS,
            glfw.KEY_EQUAL,
            glfw.KEY_KP_SUBTRACT,
            glfw.KEY_KP_ADD,
            glfw.KEY_U,
            glfw.KEY_O,
            glfw.KEY_J,
            glfw.KEY_L,
            glfw.KEY_I,
            glfw.KEY_K,
            glfw.KEY_LEFT,
            glfw.KEY_RIGHT,
            glfw.KEY_DOWN,
            glfw.KEY_UP,
            glfw.KEY_PAGE_UP,
            glfw.KEY_PAGE_DOWN,
        }

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
            self._last_keyboard_update_time = None

    def apply_camera_state(self, cam: mujoco.MjvCamera, state: CameraState) -> None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.distance = float(max(0.05, state.distance))
        cam.azimuth = float(state.azimuth)
        cam.elevation = float(state.elevation)
        self._camera_lookat_view(cam)[:] = np.asarray(state.lookat, dtype=np.float64)

    def capture_camera_state(self, cam: mujoco.MjvCamera) -> CameraState:
        return CameraState(
            distance=float(cam.distance),
            azimuth=float(cam.azimuth),
            elevation=float(cam.elevation),
            lookat=self._camera_lookat_view(cam).copy(),
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
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.azimuth += float(azimuth_delta)
        cam.elevation += float(elevation_delta)
        cam.distance = float(max(0.05, cam.distance + distance_delta))
        if lookat_delta is not None:
            lookat = self._camera_lookat_view(cam)
            lookat[:] = lookat + np.asarray(lookat_delta, dtype=np.float64)

    def format_camera_lookat(self, cam: mujoco.MjvCamera, precision: int = 6) -> str:
        return self._format_vec(self._camera_lookat_view(cam), precision=precision)

    def update_keyboard_camera_controls(self) -> None:
        if self.viewer is None or self.viewer.window is None:
            self._last_keyboard_update_time = None
            return

        now = time.monotonic()
        if self._last_keyboard_update_time is None:
            self._last_keyboard_update_time = now
            return

        dt = min(now - self._last_keyboard_update_time, 0.1)
        self._last_keyboard_update_time = now

        speed_scale = self._keyboard_speed_scale()
        angle_delta = self.angle_speed_deg * dt * speed_scale
        distance_delta = self.distance_speed * dt * speed_scale
        lookat_delta = np.zeros(3, dtype=np.float64)
        lookat_step = self.lookat_speed * dt * speed_scale

        azimuth_delta = 0.0
        elevation_delta = 0.0
        zoom_delta = 0.0

        if self._is_pressed(glfw.KEY_COMMA):
            azimuth_delta -= angle_delta
        if self._is_pressed(glfw.KEY_PERIOD):
            azimuth_delta += angle_delta
        if self._is_pressed(glfw.KEY_LEFT_BRACKET):
            elevation_delta -= angle_delta
        if self._is_pressed(glfw.KEY_RIGHT_BRACKET):
            elevation_delta += angle_delta

        if self._is_pressed(glfw.KEY_MINUS) or self._is_pressed(glfw.KEY_KP_SUBTRACT):
            zoom_delta += distance_delta
        if self._is_pressed(glfw.KEY_EQUAL) or self._is_pressed(glfw.KEY_KP_ADD):
            zoom_delta -= distance_delta

        if self._is_pressed(glfw.KEY_U) or self._is_pressed(glfw.KEY_LEFT):
            lookat_delta[0] -= lookat_step
        if self._is_pressed(glfw.KEY_O) or self._is_pressed(glfw.KEY_RIGHT):
            lookat_delta[0] += lookat_step
        if self._is_pressed(glfw.KEY_J) or self._is_pressed(glfw.KEY_DOWN):
            lookat_delta[1] -= lookat_step
        if self._is_pressed(glfw.KEY_L) or self._is_pressed(glfw.KEY_UP):
            lookat_delta[1] += lookat_step
        if self._is_pressed(glfw.KEY_I) or self._is_pressed(glfw.KEY_PAGE_UP):
            lookat_delta[2] += lookat_step
        if self._is_pressed(glfw.KEY_K) or self._is_pressed(glfw.KEY_PAGE_DOWN):
            lookat_delta[2] -= lookat_step

        if azimuth_delta or elevation_delta or zoom_delta or np.any(lookat_delta):
            self.adjust_camera(
                azimuth_delta=azimuth_delta,
                elevation_delta=elevation_delta,
                distance_delta=zoom_delta,
                lookat_delta=lookat_delta,
            )

    def _keyboard_speed_scale(self) -> float:
        scale = 1.0
        if self._is_pressed(glfw.KEY_LEFT_SHIFT) or self._is_pressed(glfw.KEY_RIGHT_SHIFT):
            scale *= self.fast_multiplier
        if self._is_pressed(glfw.KEY_LEFT_CONTROL) or self._is_pressed(glfw.KEY_RIGHT_CONTROL):
            scale *= self.slow_multiplier
        return scale

    def _is_pressed(self, key: int) -> bool:
        return self.viewer is not None and self.viewer.is_key_pressed(key)

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
    def _camera_lookat_view(cam: mujoco.MjvCamera) -> np.ndarray:
        ptr = int(cam.lookat.__array_interface__["data"][0])
        return np.ctypeslib.as_array((ctypes.c_double * 3).from_address(ptr))

    @staticmethod
    def _format_vec(vec: np.ndarray, precision: int = 6) -> str:
        values = np.asarray(vec, dtype=np.float64).reshape(-1)
        return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


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
    parser.add_argument(
        "--angle-speed",
        type=float,
        default=75.0,
        help="Keyboard azimuth/elevation speed in degrees per second.",
    )
    parser.add_argument(
        "--distance-speed",
        type=float,
        default=1.2,
        help="Keyboard zoom speed in MuJoCo distance units per second.",
    )
    parser.add_argument(
        "--lookat-speed",
        type=float,
        default=0.45,
        help="Keyboard lookat speed in MuJoCo world units per second.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tuner = InteractiveCameraTuner(env_id=args.env, seed=args.seed)
    tuner.angle_speed_deg = float(args.angle_speed)
    tuner.distance_speed = float(args.distance_speed)
    tuner.lookat_speed = float(args.lookat_speed)
    try:
        tuner.run()
    finally:
        tuner.close()


if __name__ == "__main__":
    main()
