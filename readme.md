# Panda 地铁侧窗玻璃装调

本仓库只实现一条固定的地铁侧窗玻璃装调流程：

```text
approach → descend → grasp → lift → reorient
→ coarse_align → fine_align → insert → hold
```

其中 `fine_align` 由 SAC 策略控制，其余八个阶段均为脚本控制。精定位成功后，环境才允许 pipeline 调用脚本插入和保持；RL 的 `step()` 不执行插入。

## 精定位接口

正式环境 ID 只有：

```text
FrankaWindowFineAlignDense-v0
```

- observation：11 维 `float32` Box，由 6 个归一化装调误差和上一时刻 5 维动作组成，无图像输入。
- action：5 维 `float32` Box，范围 `[-1, 1]`。
- action 0/1：窗框平面内两个平移。
- action 2/3：绕窗框平面内两轴的倾斜。
- action 4：绕窗框法向的 yaw。
- 法向间隙由环境保持在预插入平面，不属于 RL 动作。

核心分层：

```text
panda_env.py                    Panda/MuJoCo 底层控制
window_assembly_scene.py        窗框、玻璃、吸盘、碰撞与状态
window_assembly_geometry.py     装调几何和路径计算
window_assembly_controller.py   八个脚本阶段使用的运动控制
window_fine_align.py            5 DoF 精定位 MDP
window_assembly_pipeline.py     严格九阶段编排与统计
```

## 安装

```bash
pip install -r requirements.txt
```

无桌面离屏渲染时显式设置：

```bash
export MUJOCO_GL=egl
```

## 训练

单 seed：

```bash
/home/lsy/.conda/envs/panda_rl/bin/python \
  train/train_window_fine_align_sac.py \
  --timesteps 300000 \
  --seed 0 \
  --n-envs 1 \
  --device cuda:0
```

多 seed、多 GPU：

```bash
/home/lsy/.conda/envs/panda_rl/bin/python \
  launch_multi_seed.py \
  --seeds 0 1 2 3 \
  --gpus 0,1,2,3 \
  --timesteps 300000 \
  --n-envs 1
```

## 完整九阶段评估

```bash
MUJOCO_GL=egl /home/lsy/.conda/envs/panda_rl/bin/python \
  evaluate/evaluate_window_assembly_pipeline.py \
  --model outputs/window_fine_align/RUN/models/best_model.zip \
  --episodes 20 \
  --seed 0 \
  --deterministic \
  --record-video
```

评估不会从模型目录推断环境配置。若训练时改过装调参数，评估时必须显式传入相同参数。

## 阶段截图和相机

保存起始图、九个阶段末图、13 维机器人状态和汇总 JSON：

```bash
MUJOCO_GL=egl /home/lsy/.conda/envs/panda_rl/bin/python \
  capture_stage_snapshots.py \
  --model outputs/window_fine_align/RUN/models/best_model.zip \
  --seed 0 \
  --output-dir outputs/stage_snapshots/seed-0
```

截图相机可用 `--camera-distance`、`--camera-azimuth`、`--camera-elevation` 和 `--camera-lookat X Y Z` 调节。桌面环境可交互调节：

```bash
/home/lsy/.conda/envs/panda_rl/bin/python \
  capture_stage_snapshots.py --tune-camera --seed 0
```

## 验收

```bash
/home/lsy/.conda/envs/panda_rl/bin/python -m compileall -q \
  panda_mujoco_gym train evaluate utils \
  capture_stage_snapshots.py launch_multi_seed.py

/home/lsy/.conda/envs/panda_rl/bin/python -m pytest -q test tests
```

训练产物使用 flat 11 维 observation 和 5 维 action。满足这两个 space 的现有最终精定位 SAC 模型可直接显式加载；模型文件旁的环境配置不会被自动读取。
