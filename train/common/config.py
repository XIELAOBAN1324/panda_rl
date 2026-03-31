"""
学習设置类（针对 goal-conditioned sparse 任务做了增强）
"""

import os
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
import torch


def _recommended_n_envs() -> int:
    """保守的默认并行环境数。"""
    cpu_threads = os.cpu_count() or 8
    return max(8, min(32, cpu_threads // 8 or 1))


@dataclass
class BaseConfig:
    """基础配置"""

    # 环境设置
    env_name: str = "FrankaSlideDense-v0"
    algorithm: str = "SAC"

    # 训练设置
    total_timesteps: int = 1_000_000
    normalize_env: bool = True
    reward_scale: Optional[float] = None

    # 评估/保存设置
    enable_eval_callback: bool = True
    eval_freq: int = 20_000
    n_eval_episodes: int = 5
    eval_deterministic: bool = True
    enable_checkpoint_callback: bool = True
    checkpoint_freq: int = 200_000
    save_replay_buffer_checkpoints: bool = False
    save_vecnormalize_checkpoints: bool = True

    # 日志/进度
    progress_bar: bool = False
    log_interval: int = 100
    print_every_episodes: int = 20
    csv_flush_every: int = 100

    # 目录
    base_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs",
    )
    experiment_name: Optional[str] = None

    # 随机种子
    seed: Optional[int] = None

    # runtime/hardware
    device: str = "auto"
    torch_num_threads: int = 8
    torch_num_interop_threads: int = 2
    enable_tf32: bool = True
    vec_env_start_method: str = "forkserver"

    # 分阶段保存
    save_stage_models: bool = True

    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = (
                f"{self.env_name}_{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        if self.reward_scale is None:
            self.reward_scale = 1.0 if "Sparse" in self.env_name else 0.1

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            print(f"🎯 固定随机种子: {self.seed}")
        else:
            print("🎲 使用随机种子（每次 reset 初始状态会不同）")

        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")

    def create_directories(self):
        dirs_to_create = [self.exp_dir, self.model_dir, self.log_dir, self.checkpoint_dir]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 创建目录: {dir_path}")


@dataclass
class SACConfig(BaseConfig):
    """SAC 专用配置（对 sparse goal-conditioned 任务增加 HER 支持）"""

    # 向量环境数
    n_envs: int = field(default_factory=_recommended_n_envs)

    # SAC 超参数
    learning_rate: float = 1e-3
    buffer_size: int = 1_000_000
    batch_size: int = 1024
    tau: float = 0.05
    gamma: float = 0.95
    learning_starts: int = 5_000
    train_freq: int = 1
    gradient_steps: int = 1

    # 网络结构
    policy_width: int = 256
    policy_depth: int = 2
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": [256, 256],
        "activation_fn": torch.nn.ReLU,
        "normalize_images": False,
    })

    # 探索
    action_noise_std: float = 0.2
    use_sde: bool = True
    sde_sample_freq: int = 8

    # HER
    use_her: Optional[bool] = None
    her_n_sampled_goal: int = 4
    her_goal_selection_strategy: str = "future"

    # 从 dense checkpoint 初始化（仅复制网络权重）
    init_model_path: Optional[str] = None

    # 训练阶段定义（录像用）
    stages: Dict[str, float] = field(default_factory=lambda: {
        "0_random": 0.0,
        "1_20percent": 0.2,
        "2_40percent": 0.4,
        "3_60percent": 0.6,
        "4_80percent": 0.8,
        "5_100percent": 1.0,
    })

    def __post_init__(self):
        self.policy_kwargs = {
            "net_arch": [self.policy_width] * self.policy_depth,
            "activation_fn": torch.nn.ReLU,
            "normalize_images": False,
        }

        if self.use_her is None:
            self.use_her = "Sparse" in self.env_name

        super().__post_init__()

    def get_stage_timesteps(self) -> Dict[str, int]:
        return {name: int(ratio * self.total_timesteps) for name, ratio in self.stages.items()}
