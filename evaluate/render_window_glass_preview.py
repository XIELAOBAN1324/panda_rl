from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "panda_mujoco_gym" / "assets" / "pick_and_place_window.xml"
OUTPUT_PATH = REPO_ROOT / "outputs" / "window_glass_preview" / "window_glass_preview.png"

NEUTRAL_QPOS = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79], dtype=np.float64)
WINDOW_CENTER = np.array([0.74288041, -0.29686477, 0.65], dtype=np.float64)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    data.qpos[: NEUTRAL_QPOS.size] = NEUTRAL_QPOS
    data.ctrl[:7] = NEUTRAL_QPOS

    window_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "window_frame")
    model.body_pos[window_body_id] = WINDOW_CENTER

    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "watching")
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = cam_id

    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    Image.fromarray(image).save(OUTPUT_PATH)
    print(f"Saved preview to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
