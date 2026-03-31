#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

FILES = {
    "train/common/config.py": r'''"""학습 설정 클래스 (수렴 안정성 우선판)."""

import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch


def _recommended_n_envs() -> int:
    """MuJoCo + SB3 SAC 에서 과도한 병렬화로 인한 샘플 효율 저하를 피하기 위한 보수적 기본값."""
    cpu_threads = os.cpu_count() or 8
    return max(1, min(8, cpu_threads // 4))


@dataclass
class BaseConfig:
    """기본 설정 클래스."""

    env_name: str = "FrankaSlideDense-v0"
    algorithm: str = "SAC"

    total_timesteps: int = 1_000_000
    normalize_env: bool = True
    reward_scale: float = 1.0

    enable_eval_callback: bool = True
    eval_freq: int = 50_000
    n_eval_episodes: int = 3
    eval_deterministic: bool = True
    enable_checkpoint_callback: bool = True
    checkpoint_freq: int = 250_000
    save_replay_buffer_checkpoints: bool = False
    save_vecnormalize_checkpoints: bool = True

    progress_bar: bool = False
    log_interval: int = 100
    print_every_episodes: int = 20
    csv_flush_every: int = 100

    base_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs",
    )
    experiment_name: Optional[str] = None

    seed: Optional[int] = None

    device: str = "auto"
    torch_num_threads: int = 8
    torch_num_interop_threads: int = 2
    enable_tf32: bool = True
    vec_env_start_method: str = "forkserver"

    save_stage_models: bool = True

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.env_name}_{self.algorithm}_{timestamp}"

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            print(f"🎲 시드 고정: {self.seed}")
        else:
            print("🎲 랜덤 시드 사용 (매 에피소드마다 다른 초기 상태)")

        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")

    def create_directories(self) -> None:
        for dir_path in [self.exp_dir, self.model_dir, self.log_dir, self.checkpoint_dir]:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 디렉토리 생성: {dir_path}")


@dataclass
class SACConfig(BaseConfig):
    """SAC 전용 설정 (샘플 효율/수렴 안정성 중심)."""

    n_envs: int = field(default_factory=_recommended_n_envs)

    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.98
    learning_starts: int = 10_000
    train_freq: int = 1
    gradient_steps: int = 1

    policy_width: int = 256
    policy_depth: int = 2
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": [256, 256],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
    })

    action_noise_std: float = 0.1
    use_sde: bool = True
    sde_sample_freq: int = 8

    stages: Dict[str, float] = field(default_factory=lambda: {
        "0_random": 0.0,
        "1_20percent": 0.2,
        "2_40percent": 0.4,
        "3_60percent": 0.6,
        "4_80percent": 0.8,
        "5_100percent": 1.0,
    })

    def __post_init__(self) -> None:
        self.policy_kwargs = {
            "net_arch": [self.policy_width] * self.policy_depth,
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,
        }
        super().__post_init__()

    def get_stage_timesteps(self) -> Dict[str, int]:
        return {
            name: int(ratio * self.total_timesteps)
            for name, ratio in self.stages.items()
        }
''',
    "train/train_sac.py": r'''#!/usr/bin/env python3
"""SAC 학습 스크립트 (수렴 안정성 우선판)."""

import argparse
import json
import os
import sys
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
    sync_envs_normalization,
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import panda_mujoco_gym
from train.common.callbacks import TrainingCallback
from train.common.config import SACConfig, _recommended_n_envs
from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper


def configure_runtime(config: SACConfig) -> None:
    """Torch/CUDA 런타임 튜닝."""
    if config.torch_num_threads is not None:
        torch.set_num_threads(int(config.torch_num_threads))
    if config.torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(config.torch_num_interop_threads))
        except RuntimeError:
            pass

    if config.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("⚙️ 런타임 설정")
    print(f"   device: {config.device}")
    print(f"   torch_num_threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"   cuda available: True ({torch.cuda.device_count()} GPU)")
        print(f"   TF32 enabled: {bool(config.enable_tf32)}")
    else:
        print("   cuda available: False")


def create_env(
    env_name: str,
    render_mode: Optional[str] = None,
    reward_scale: float = 1.0,
    seed: Optional[int] = None,
):
    """학습/평가에서 공통으로 쓰는 단일 환경 생성."""
    env = gym.make(env_name, render_mode=render_mode)

    # 등록된 max_episode_steps(TimeLimit)를 그대로 사용하고,
    # reward/logging 기준을 맞추기 위해 scaling 후 Monitor 를 적용한다.
    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)
    env = Monitor(env)
    env = SuccessTrackingWrapper(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


def create_vec_env(
    env_name: str,
    n_envs: int = 1,
    normalize: bool = True,
    reward_scale: float = 1.0,
    seed: Optional[int] = None,
    vec_normalize_path: Optional[str] = None,
    training: bool = True,
    render_mode: Optional[str] = None,
    start_method: str = "forkserver",
):
    """벡터화 환경 생성."""

    def make_env(rank: int):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_env(
                env_name,
                render_mode=render_mode,
                reward_scale=reward_scale,
                seed=env_seed,
            )

        return _init

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        vec_env = SubprocVecEnv(
            [make_env(i) for i in range(n_envs)],
            start_method=start_method,
        )

    if normalize:
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        vec_env.training = training
        vec_env.norm_reward = False

    return vec_env


def create_sac_model(env, config: SACConfig):
    """SAC 모델 생성."""
    n_actions = env.action_space.shape[-1]
    action_noise = None
    if not config.use_sde:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.action_noise_std * np.ones(n_actions),
        )

    return SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        action_noise=action_noise,
        policy_kwargs=config.policy_kwargs,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        verbose=1,
        tensorboard_log=config.log_dir,
        device=config.device,
    )


def _to_callback_frequency(target_steps: int, n_envs: int) -> int:
    return max(target_steps // max(n_envs, 1), 1)


def _make_summary(config, training_time, mean_reward, std_reward, training_callback):
    serializable_config = {}
    for k, v in config.__dict__.items():
        try:
            json.dumps({k: v})
            serializable_config[k] = v
        except TypeError:
            serializable_config[k] = repr(v)

    return {
        "experiment_name": config.experiment_name,
        "env_name": config.env_name,
        "algorithm": "SAC",
        "total_timesteps": config.total_timesteps,
        "training_time_hours": training_time / 3600,
        "final_mean_reward": float(mean_reward),
        "final_std_reward": float(std_reward),
        "best_reward": float(training_callback.best_reward),
        "final_success_rate": float(training_callback.recent_success_rate),
        "total_episodes": int(training_callback.episode_count),
        "config": serializable_config,
    }


def _save_summary(summary, *paths):
    for summary_path in paths:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"📝 학습 요약 저장: {summary_path}")


def train_sac(config: SACConfig):
    configure_runtime(config)
    print("🚀 SAC 학습 시작!")
    print(f"   환경: {config.env_name}")
    print(f"   총 학습 스텝: {config.total_timesteps:,}")
    print(f"   n_envs: {config.n_envs}")
    print(f"   start_method: {config.vec_env_start_method}")
    print(f"   reward_scale: {config.reward_scale}")
    print(f"   결과 저장 위치: {config.exp_dir}")
    print("-" * 60)

    config.create_directories()

    print("🛠️ 환경 생성 중...")
    env = create_vec_env(
        config.env_name,
        n_envs=config.n_envs,
        normalize=config.normalize_env,
        reward_scale=config.reward_scale,
        seed=config.seed,
        start_method=config.vec_env_start_method,
        training=True,
    )

    eval_env = None
    if config.enable_eval_callback and config.n_eval_episodes > 0:
        eval_env = create_vec_env(
            config.env_name,
            n_envs=1,
            normalize=config.normalize_env,
            reward_scale=config.reward_scale,
            seed=config.seed,
            start_method=config.vec_env_start_method,
            training=False,
        )

    logger_path = os.path.join(config.log_dir, "tensorboard")
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])

    print("\n🧠 SAC 모델 초기화...")
    model = create_sac_model(env, config)
    model.set_logger(new_logger)

    eval_freq = _to_callback_frequency(config.eval_freq, config.n_envs)
    checkpoint_freq = _to_callback_frequency(config.checkpoint_freq, config.n_envs)
    print(
        f"🔁 평가 주기: {config.eval_freq} env-steps "
        f"(callback step {eval_freq}, n_envs={config.n_envs})"
    )
    print(
        f"💾 체크포인트 주기: {config.checkpoint_freq} env-steps "
        f"(callback step {checkpoint_freq}, n_envs={config.n_envs})"
    )

    callbacks = [TrainingCallback(config)]

    if config.enable_eval_callback and eval_env is not None and config.n_eval_episodes > 0:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=config.model_dir,
                log_path=os.path.join(config.log_dir, "eval"),
                eval_freq=eval_freq,
                n_eval_episodes=config.n_eval_episodes,
                deterministic=config.eval_deterministic,
                verbose=1,
            )
        )

    if config.enable_checkpoint_callback:
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=config.checkpoint_dir,
                name_prefix=f"sac_{config.env_name}",
                save_replay_buffer=config.save_replay_buffer_checkpoints,
                save_vecnormalize=config.save_vecnormalize_checkpoints,
            )
        )

    print("\n📚 학습 시작!")
    print("=" * 60)
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            progress_bar=config.progress_bar,
        )
    except KeyboardInterrupt:
        print("\n⏸️ 학습이 중단되었습니다.")
    end_time = time.time()

    training_time = end_time - start_time
    print("\n✅ 학습 완료!")
    print(f"⏱️ 총 학습 시간: {training_time / 3600:.2f}시간")

    final_model_path = os.path.join(config.model_dir, "final_model")
    model.save(final_model_path)
    if hasattr(env, "save"):
        env.save(os.path.join(config.model_dir, "vec_normalize.pkl"))
    print(f"💾 최종 모델 저장: {final_model_path}")

    if eval_env is None:
        eval_env = create_vec_env(
            config.env_name,
            n_envs=1,
            normalize=config.normalize_env,
            reward_scale=config.reward_scale,
            seed=config.seed,
            start_method=config.vec_env_start_method,
            training=False,
        )

    if isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
        sync_envs_normalization(env, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    print("\n🔍 최종 평가...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )
    print(f"🏆 최종 평가 결과: {mean_reward:.2f} ± {std_reward:.2f}")

    summary = _make_summary(config, training_time, mean_reward, std_reward, callbacks[0])
    _save_summary(
        summary,
        os.path.join(config.exp_dir, "training_summary.json"),
        os.path.join(config.log_dir, "training_summary.json"),
    )

    return model, env


def main():
    parser = argparse.ArgumentParser(description="SAC 학습 스크립트 (수렴 안정성 우선판)")
    parser.add_argument("--env", type=str, default="FrankaSlideDense-v0", help="환경 이름")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="총 학습 스텝")
    parser.add_argument("--exp-name", type=str, default=None, help="실험 이름")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="보상 스케일")
    parser.add_argument("--n-envs", type=int, default=None, help="병렬 환경 개수")
    parser.add_argument("--seed", type=int, default=None, help="난수 시드")
    parser.add_argument("--device", type=str, default="auto", help="학습 디바이스")
    parser.add_argument("--torch-threads", type=int, default=8, help="torch intra-op threads")
    parser.add_argument("--torch-interop-threads", type=int, default=2, help="torch inter-op threads")
    parser.add_argument(
        "--vec-start-method",
        type=str,
        default="forkserver",
        choices=["fork", "forkserver", "spawn"],
        help="SubprocVecEnv start method",
    )
    parser.add_argument("--eval-freq", type=int, default=50_000, help="학습 중 평가 주기")
    parser.add_argument("--n-eval-episodes", type=int, default=3, help="학습 중 평가 에피소드 수")
    parser.add_argument("--checkpoint-freq", type=int, default=250_000, help="체크포인트 저장 주기")
    parser.add_argument("--save-replay-buffer", action="store_true", help="체크포인트에 replay buffer 저장")
    parser.add_argument("--no-eval", action="store_true", help="학습 중 평가 비활성화")
    parser.add_argument("--progress-bar", action="store_true", help="progress bar 표시")
    parser.add_argument("--policy-width", type=int, default=256, help="정책/가치망 hidden width")
    parser.add_argument("--policy-depth", type=int, default=2, help="정책/가치망 hidden depth")
    parser.add_argument("--batch-size", type=int, default=256, help="SAC batch size")
    parser.add_argument("--no-tf32", action="store_true", help="Ampere 이상 GPU 의 TF32 비활성화")
    args = parser.parse_args()

    config = SACConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        experiment_name=args.exp_name,
        reward_scale=args.reward_scale,
        n_envs=args.n_envs if args.n_envs is not None else _recommended_n_envs(),
        seed=args.seed,
        device=args.device,
        torch_num_threads=args.torch_threads,
        torch_num_interop_threads=args.torch_interop_threads,
        vec_env_start_method=args.vec_start_method,
        enable_eval_callback=not args.no_eval,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        checkpoint_freq=args.checkpoint_freq,
        save_replay_buffer_checkpoints=args.save_replay_buffer,
        progress_bar=args.progress_bar,
        policy_width=args.policy_width,
        policy_depth=args.policy_depth,
        batch_size=args.batch_size,
        enable_tf32=not args.no_tf32,
    )
    train_sac(config)

    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print(f"📁 결과 저장 위치: {config.exp_dir}")
    print("📌 다음 단계:")
    print(f"python evaluate/evaluate_with_video.py --exp-dir {config.exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
''',
    "utils/env_utils.py": r'''"""환경 생성 유틸리티 함수들."""

import os
import sys
from typing import Optional

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper


def create_env(
    env_name,
    render_mode=None,
    reward_scale: float = 1.0,
    seed: Optional[int] = None,
):
    """환경 생성 (래퍼 적용)."""
    env = gym.make(env_name, render_mode=render_mode)
    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)
    env = Monitor(env)
    env = SuccessTrackingWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def create_vec_env(
    env_name,
    n_envs=1,
    normalize=True,
    reward_scale: float = 1.0,
    vec_normalize_path=None,
    training=True,
    render_mode=None,
    seed: Optional[int] = None,
    start_method: str = "forkserver",
):
    """벡터화된 환경 생성."""

    def make_env(rank: int):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_env(
                env_name,
                render_mode=render_mode,
                reward_scale=reward_scale,
                seed=env_seed,
            )

        return _init

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        vec_env = SubprocVecEnv(
            [make_env(i) for i in range(n_envs)],
            start_method=start_method,
        )

    if normalize:
        if vec_normalize_path and os.path.exists(vec_normalize_path):
            vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)
        vec_env.training = training
        vec_env.norm_reward = False

    return vec_env
''',
    "panda_mujoco_gym/envs/panda_env.py": r'''import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Any, Optional, SupportsFloat

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}


class FrankaEnv(MujocoRobotEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: str = None,
        n_substeps: int = 50,
        reward_type: str = "sparse",
        block_gripper: bool = False,
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        obj_xy_range: float = 0.3,
        goal_x_offset: float = 0.4,
        goal_z_range: float = 0.2,
        **kwargs,
    ):
        self.block_gripper = block_gripper
        self.model_path = model_path
        action_size = 3 + (0 if self.block_gripper else 1)
        self.reward_type = reward_type
        self.neutral_joint_values = np.array(
            [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00]
        )

        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.distance_threshold = distance_threshold
        self.obj_xy_range = obj_xy_range
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        self.goal_range_low = np.array(
            [-self.goal_xy_range / 2 + goal_x_offset, -self.goal_xy_range / 2, 0.0]
        )
        self.goal_range_high = np.array(
            [self.goal_xy_range / 2 + goal_x_offset, self.goal_xy_range / 2, self.goal_z_range]
        )
        self.obj_range_low = np.array([-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0.0])
        self.obj_range_high = np.array([self.obj_xy_range / 2, self.obj_xy_range / 2, 0.0])

        self.goal_range_low[0] += 0.6
        self.goal_range_high[0] += 0.6
        self.obj_range_low[0] += 0.6
        self.obj_range_high[0] += 0.6

        self.nu = self.model.nu
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.ctrl_range = self.model.actuator_ctrlrange

    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)
        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        free_joint_index = self._model_names.joint_names.index("obj_joint")
        self.arm_joint_names = self._model_names.joint_names[:free_joint_index][0:7]
        self.gripper_joint_names = self._model_names.joint_names[:free_joint_index][7:9]

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(
            self.model, self.data, "ee_center_site"
        ).copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)
        self._mujoco_step()
        self.initial_object_height = self._utils.get_joint_qpos(
            self.model, self.data, "obj_joint"
        )[2].copy()

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
        goal_distance = float(self.goal_distance(obs["achieved_goal"], self.goal))
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "goal_distance": goal_distance,
            "ee_object_distance": float(
                np.linalg.norm(self.get_ee_position().copy() - obs["achieved_goal"])
            ),
            "object_height": float(obs["achieved_goal"][2]),
        }
        terminated = bool(info["is_success"])
        truncated = bool(self.compute_truncated(obs["achieved_goal"], self.goal, info))
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)

        reward = -d

        # HER / 벡터 입력에서는 기본 거리 보상만 유지한다.
        if np.ndim(d) != 0:
            return reward

        object_position = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_object_distance = float(np.linalg.norm(ee_position - object_position))
        reward -= 0.25 * ee_object_distance

        if not self.block_gripper:
            lift_height = max(0.0, float(object_position[2] - self.initial_object_height))
            reward += 1.5 * lift_height
            if lift_height > 0.02:
                reward += 0.25
            if d < self.distance_threshold:
                reward += 1.0
        elif d < self.distance_threshold:
            reward += 0.5

        return np.float32(reward)

    def _set_action(self, action) -> None:
        action = action.copy()
        if not self.block_gripper:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(
                fingers_width / 2,
                self.ctrl_range[-1, 0],
                self.ctrl_range[-1, 1],
            )
        else:
            pos_ctrl = action
            fingers_half_width = 0.0

        self.data.ctrl[-2:] = fingers_half_width

        pos_ctrl *= 0.05
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = np.max((0.0, pos_ctrl[2]))
        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)

    def _get_obs(self) -> dict:
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        ee_velocity = (
            self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy()
            * self.dt
        )
        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()

        object_position = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()
        object_rotation = rotations.mat2euler(
            self._utils.get_site_xmat(self.model, self.data, "obj_site")
        ).copy()
        object_velp = self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy() * self.dt
        object_velr = self._utils.get_site_xvelr(self.model, self.data, "obj_site").copy() * self.dt

        if not self.block_gripper:
            observation = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    fingers_width,
                    object_position,
                    object_rotation,
                    object_velp,
                    object_velr,
                ]
            ).copy()
        else:
            observation = np.concatenate(
                [
                    ee_position,
                    ee_velocity,
                    object_position,
                    object_rotation,
                    object_velp,
                    object_velr,
                ]
            ).copy()

        return {
            "observation": observation,
            "achieved_goal": object_position.copy(),
            "desired_goal": self.goal.copy(),
        }

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self) -> None:
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
        # 이전 버전은 action 당 10번 반복되어 실제 physics 가 10배 느려졌다.
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._mujoco.mj_forward(model, data)

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def set_joint_neutral(self) -> None:
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

    def _sample_goal(self) -> np.ndarray:
        goal = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if not self.block_gripper and self.goal_z_range > 0.0:
            # 지나치게 많은 공중 목표는 초반 탐색을 어렵게 만든다.
            if self.np_random.random() < 0.7:
                noise[2] = 0.0
        goal += noise
        return goal

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

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_body_state(self, name) -> np.ndarray:
        body_id = self._model_names.body_name2id[name]
        body_xpos = self.data.xpos[body_id]
        body_xquat = self.data.xquat[body_id]
        return np.concatenate([body_xpos, body_xquat])

    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2
''',
    "evaluate/evaluate_with_video.py": r'''#!/usr/bin/env python3
"""모델 평가 및 비디오 생성 통합 스크립트."""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import panda_mujoco_gym
from utils import create_vec_env, find_latest_experiment, get_experiment_info, load_model, save_json
from video_recorder import StageVideoRecorder


def _predict_action(model, obs, env):
    if model is None:
        raw_action = env.action_space.sample()
        return np.array([raw_action])
    action, _ = model.predict(obs, deterministic=True)
    return action


def evaluate_model_performance(model, env, num_episodes: int = 50) -> Dict:
    """모델 또는 랜덤 정책 성능 평가.

    evaluate_policy 와 수동 rollout 을 섞지 않고, 동일 에피소드에서 reward/length/success 를 함께 집계한다.
    """
    policy_name = "랜덤 정책" if model is None else "모델"
    print(f"\n📊 {policy_name} 성능 평가 중... ({num_episodes}개 에피소드)")

    rewards = []
    lengths = []
    success_count = 0

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        length = 0

        while not done:
            action = _predict_action(model, obs, env)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            total_reward += float(reward[0])
            length += 1
            if done and infos[0].get("is_success", False):
                success_count += 1

        rewards.append(total_reward)
        lengths.append(length)

    results = {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": success_count / num_episodes,
        "num_episodes": num_episodes,
        "all_rewards": [float(r) for r in rewards],
        "all_lengths": [int(l) for l in lengths],
    }

    print(f"   평균 보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   성공률: {results['success_rate']:.3f}")
    print(f"   평균 에피소드 길이: {results['mean_length']:.1f}")
    return results


def _get_vecnormalize_path(models_dir: str, model_name: str):
    stage_vecnorm = os.path.join(models_dir, f"{model_name}_vecnormalize.pkl")
    final_vecnorm = os.path.join(models_dir, "vec_normalize.pkl")
    if os.path.exists(stage_vecnorm):
        return stage_vecnorm
    if os.path.exists(final_vecnorm):
        return final_vecnorm
    return None


def _build_stage_model_list(available_models: List[str], stages: List[str] = None) -> List[str]:
    if stages is None or "all" in stages:
        selected = [m for m in available_models if m.startswith("stage_")]
    else:
        selected = []
        for stage in stages:
            model_name = stage if stage.startswith("stage_") else f"stage_{stage}"
            if model_name in available_models:
                selected.append(model_name)
        if "final" in stages and "final_model" in available_models:
            selected.append("final_model")

    if "final_model" in available_models and (stages is None or "all" in stages):
        selected.append("final_model")

    def sort_key(name: str):
        if name == "final_model":
            return (999, name)
        if name.startswith("stage_"):
            stage_label = name.replace("stage_", "", 1)
            try:
                stage_num = int(stage_label.split("_", 1)[0])
            except ValueError:
                stage_num = 998
            return (stage_num, name)
        return (997, name)

    return sorted(dict.fromkeys(selected), key=sort_key)


def evaluate_experiment(
    exp_dir: str,
    stages: List[str] = None,
    num_eval_episodes: int = 50,
    num_video_episodes: int = 3,
    record_video: bool = True,
    create_highlights: bool = True,
) -> Dict:
    print("=" * 60)
    print("🧪 실험 평가 시작")
    print(f"📁 실험 디렉토리: {exp_dir}")
    print("=" * 60)

    exp_info = get_experiment_info(exp_dir)
    if not exp_info:
        raise ValueError(f"실험 정보를 찾을 수 없습니다: {exp_dir}")

    env_name = exp_info.get("env_name", "FrankaSlideDense-v0")
    algorithm = exp_info.get("algorithm", "SAC")
    reward_scale = exp_info.get("config", {}).get("reward_scale", 1.0)

    print("\n📌 실험 정보:")
    print(f"   환경: {env_name}")
    print(f"   알고리즘: {algorithm}")
    print(f"   보상 스케일: {reward_scale}")
    print(f"   총 학습 스텝: {exp_info.get('total_timesteps', 'Unknown')}")

    models_dir = os.path.join(exp_dir, "models")
    available_models = exp_info.get("available_models", [])
    if not available_models:
        raise ValueError(f"모델을 찾을 수 없습니다: {models_dir}")

    stage_models = _build_stage_model_list(available_models, stages)
    print(f"\n📋 평가할 모델 ({len(stage_models)}개):")
    for model_name in stage_models:
        print(f"   - {model_name}")

    evaluation_dir = os.path.join(exp_dir, "evaluation")
    os.makedirs(evaluation_dir, exist_ok=True)

    video_recorder = None
    if record_video:
        videos_dir = os.path.join(evaluation_dir, "videos")
        video_recorder = StageVideoRecorder(videos_dir, fps=10)

    all_results = {
        "experiment_info": exp_info,
        "evaluation_config": {
            "num_eval_episodes": num_eval_episodes,
            "num_video_episodes": num_video_episodes,
            "stages_evaluated": stage_models,
        },
        "model_performances": {},
        "video_recordings": {},
    }

    for model_name in stage_models:
        print(f"\n{'=' * 50}")
        print(f"🔍 평가 중: {model_name}")
        print(f"{'=' * 50}")

        model_path = os.path.join(models_dir, f"{model_name}.zip")
        vec_normalize_path = _get_vecnormalize_path(models_dir, model_name)

        eval_env = create_vec_env(
            env_name,
            n_envs=1,
            normalize=True,
            reward_scale=reward_scale,
            vec_normalize_path=vec_normalize_path,
            training=False,
            render_mode=None,
        )

        if model_name == "stage_0_random":
            model = None
            print("🎲 Random Policy 사용")
        else:
            model = load_model(model_path, algorithm=algorithm, env=eval_env)
            print("✅ 모델 로드 완료")
            if vec_normalize_path:
                print(f"📦 정규화 통계: {os.path.basename(vec_normalize_path)}")

        performance = evaluate_model_performance(model, eval_env, num_eval_episodes)
        all_results["model_performances"][model_name] = performance
        eval_env.close()

        if record_video and video_recorder is not None:
            video_env = create_vec_env(
                env_name,
                n_envs=1,
                normalize=True,
                reward_scale=reward_scale,
                vec_normalize_path=vec_normalize_path,
                training=False,
                render_mode="rgb_array",
            )
            stage_name = model_name.replace("stage_", "") if model_name.startswith("stage_") else model_name
            video_results = video_recorder.record_stage_episodes(stage_name, model, video_env, num_video_episodes)
            all_results["video_recordings"][model_name] = {
                "num_videos": len(video_results),
                "episodes": video_results,
            }
            video_env.close()

    if record_video and create_highlights and video_recorder is not None:
        video_recorder.create_highlight_reel()

    if record_video and video_recorder is not None:
        video_summary = video_recorder.save_evaluation_summary()
        all_results["video_summary"] = video_summary

    results_path = os.path.join(evaluation_dir, "evaluation_results.json")
    save_json(all_results, results_path)

    print("\n" + "=" * 60)
    print("✅ 평가 완료!")
    print(f"📊 결과 저장: {results_path}")
    if record_video:
        print(f"🎥 비디오 저장: {os.path.join(evaluation_dir, 'videos')}")
    print("=" * 60)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="모델 평가 및 비디오 생성")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--exp-dir", type=str, help="실험 디렉토리 경로")
    group.add_argument("--latest", action="store_true", help="가장 최근 실험 평가")

    parser.add_argument("--stages", type=str, nargs="+", default=None, help="평가할 단계 또는 'all'")
    parser.add_argument("--num-eval", type=int, default=50, help="성능 평가 에피소드 수")
    parser.add_argument("--num-video", type=int, default=3, help="각 단계별 비디오 에피소드 수")
    parser.add_argument("--no-video", action="store_true", help="비디오 녹화 비활성화")
    parser.add_argument("--no-highlights", action="store_true", help="하이라이트 생성 비활성화")
    args = parser.parse_args()

    if args.latest:
        exp_dir = find_latest_experiment()
        if exp_dir is None:
            print("❌ 최근 실험을 찾을 수 없습니다.")
            return
        print(f"📁 최신 실험 발견: {exp_dir}")
    else:
        exp_dir = args.exp_dir
        if not os.path.exists(exp_dir):
            print(f"❌ 실험 디렉토리를 찾을 수 없습니다: {exp_dir}")
            return

    try:
        results = evaluate_experiment(
            exp_dir=exp_dir,
            stages=args.stages,
            num_eval_episodes=args.num_eval,
            num_video_episodes=args.num_video,
            record_video=not args.no_video,
            create_highlights=not args.no_highlights,
        )

        print("\n📈 평가 결과 요약:")
        for model_name, perf in results["model_performances"].items():
            if "mean_reward" in perf:
                print(
                    f"   {model_name}: 보상 {perf['mean_reward']:.2f} ± {perf['std_reward']:.2f}, "
                    f"성공률 {perf['success_rate']:.3f}"
                )
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
''',
}


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python apply_panda_convergence_fixes.py /path/to/panda_rl")
        return 1

    repo_root = Path(sys.argv[1]).expanduser().resolve()
    if not repo_root.exists():
        print(f"Repository path not found: {repo_root}")
        return 1

    required = [repo_root / "train", repo_root / "utils", repo_root / "panda_mujoco_gym"]
    if not all(path.exists() for path in required):
        print(f"Not a valid panda_rl repo root: {repo_root}")
        return 1

    backup_dir = repo_root / f"backup_before_convergence_fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Repo root: {repo_root}")
    print(f"[INFO] Backup dir: {backup_dir}")

    for rel_path, content in FILES.items():
        target = repo_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            backup_path = backup_dir / rel_path
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_path)

        target.write_text(content + "\n", encoding="utf-8")
        print(f"[OK] Wrote {rel_path}")

    print("[DONE] Convergence-oriented fixes applied.")
    print("[NEXT] Run: git diff")
    print(
        "[NEXT] Run: python -m py_compile train/common/config.py train/train_sac.py "
        "utils/env_utils.py panda_mujoco_gym/envs/panda_env.py evaluate/evaluate_with_video.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
