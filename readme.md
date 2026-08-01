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
panda_env.py                    Panda、MuJoCo、mocap、控制步进和渲染基础能力
window_assembly_scene.py        窗框、玻璃、吸盘、场景采样、对象状态和碰撞检测
window_assembly_geometry.py     坐标系、四角点、位姿误差、投影和路径数学
window_assembly_controller.py   八个脚本阶段、底层插值、阶段诊断和最终装配验证
window_fine_align.py            精定位 reset、11 维 observation、5 维 step、reward 和终止
window_assembly_pipeline.py     九阶段编排、仅在 fine_align 调用 policy、结果统计
```

环境 reset 调用 controller 完成 `approach` 至 `coarse_align`，然后只暴露
`fine_align` MDP。精定位成功后，pipeline 直接调用 controller 的 `run_insert()`、
`run_hold()` 和 `verify_final_assembly()`。controller 不依赖训练代码，也不调用策略。

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
  --model outputs/window_fine_align/sac-20260801-183135-seed0/models/best_model.zip \
  --episodes 5 \
  --seed 0 \
  --deterministic \
  --record-video \
  --output-dir outputs/visualization/modle1500/
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
  panda_mujoco_gym train evaluate utils tests \
  capture_stage_snapshots.py launch_multi_seed.py

/home/lsy/.conda/envs/panda_rl/bin/python -m pytest -q tests
```

训练产物使用 flat 11 维 observation 和 5 维 action。满足这两个 space 的现有最终精定位 SAC 模型可直接显式加载；模型文件旁的环境配置不会被自动读取。
