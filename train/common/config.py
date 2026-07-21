"""训练配置类（pick-and-place sparse + HER 默认值，移除 curriculum 版）"""
import os
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union

import numpy as np
import torch


def _recommended_n_envs() -> int:
    """保守的默认并行环境数。"""
    cpu_threads = os.cpu_count() or 8
    return max(8, min(32, cpu_threads // 8 or 1))


@dataclass
class BaseConfig:
    """基础配置类"""

    # 环境设置
    env_name: str = "FrankaPickAndPlaceWindowDense-v0"
    algorithm: str = "SAC"

    # 训练设置
    total_timesteps: int = 1_000_000
    normalize_env: bool = True
    reward_scale: float = 0.1

    # 评估/保存设置
    enable_eval_callback: bool = True
    eval_freq: int = 20_000
    n_eval_episodes: int = 5
    eval_deterministic: bool = True
    enable_checkpoint_callback: bool = True
    checkpoint_freq: int = 200_000
    save_replay_buffer_checkpoints: bool = False
    save_vecnormalize_checkpoints: bool = True

    # 日志/进度显示
    progress_bar: bool = False
    log_interval: int = 100
    print_every_episodes: int = 20
    csv_flush_every: int = 100

    # 目录设置
    base_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "outputs",
    )
    experiment_name: Optional[str] = None

    # 随机种子
    seed: Optional[int] = None

    # 运行时/硬件设置
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

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            print(f"🎲 固定随机种子: {self.seed}")
        else:
            print("🎲 使用随机种子（每个回合的初始状态都不同）")

        self.exp_dir = os.path.join(self.base_dir, self.experiment_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")

    def create_directories(self):
        dirs_to_create = [self.exp_dir, self.model_dir, self.log_dir, self.checkpoint_dir]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 已创建目录: {dir_path}")


@dataclass
class SACConfig(BaseConfig):
    """SAC 专用配置（支持 HER，禁用 curriculum）"""

    # 向量环境并行数量
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
    ent_coef: Union[str, float] = "auto"
    target_entropy: Union[str, float] = "auto"
    min_ent_coef: Optional[float] = None

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
    her: bool = False
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"
    copy_info_dict: bool = False

    # Curriculum（向后兼容的 no-op）
    curriculum: bool = False
    curriculum_total_env_steps: Optional[int] = None
    dense_reward_shaping: bool = False
    dense_reward_style: str = "standard"
    safe_curriculum: bool = False
    task_geometry_features: bool = False
    task_stage_features: bool = False
    task_progress_features: bool = False
    residual_guidance: bool = False
    residual_action_scale: float = 0.1

    # 可选 warm-start
    init_model_path: Optional[str] = None
    expert_warmstart_only: bool = False
    expert_demo_episodes: int = 0
    expert_demo_style: str = "staged"
    expert_hint_style: str = "staged"
    bc_pretrain_epochs: int = 0
    bc_batch_size: int = 512
    bc_learning_rate: Optional[float] = None
    prefill_replay_buffer: bool = True
    demo_prefill_passes: int = 1

    # 训练阶段定义（用于视频录制）
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
        self.curriculum = False
        self.curriculum_total_env_steps = None
        super().__post_init__()

    def get_stage_timesteps(self) -> Dict[str, int]:
        return {name: int(ratio * self.total_timesteps) for name, ratio in self.stages.items()}
