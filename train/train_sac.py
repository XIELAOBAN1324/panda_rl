#!/usr/bin/env python3
"""SAC 训练脚本（以 HER 为核心的 pick-and-place sparse 默认配置，移除 curriculum 版）"""
import argparse
import json
import os
import sys
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import panda_mujoco_gym  # noqa: F401
from train.common.callbacks import TrainingCallback
from train.common.config import SACConfig, _recommended_n_envs
from train.common.curriculum import SafePickAndPlaceCurriculumWrapper
from train.common.expert_pickplace import ExpertDataset, collect_pickplace_expert_dataset
from train.common.wrappers import (
    PickAndPlaceResidualGuidanceWrapper,
    PickAndPlaceGeometryWrapper,
    PickAndPlaceStageFeatureWrapper,
    PickAndPlaceTaskProgressWrapper,
    PickAndPlaceDenseRewardWrapper,
    RewardScalingWrapper,
    SuccessTrackingWrapper,
)


CURRICULUM_REMOVED_WARNING = (
    "[WARN] Pick-and-place curriculum has been removed because it caused "
    "reward/task mismatch on sparse training. The flag is ignored and full-task "
    "training will be used."
)


def print_pickplace_sparse_effective_config(config: SACConfig) -> None:
    if not is_pick_and_place_sparse(config.env_name):
        return

    print("\n🧩 PickPlace Sparse 最终配置")
    print(f"  profile: {getattr(config, 'pickplace_profile', 'auto')}")
    print(f"  n_envs: {config.n_envs}")
    print(f"  reward_scale: {config.reward_scale}")
    print(f"  normalize_env: {config.normalize_env}")
    print(f"  her: {config.her}")
    print(f"  n_sampled_goal: {config.n_sampled_goal}")
    print(f"  goal_selection_strategy: {config.goal_selection_strategy}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_starts: {config.learning_starts}")
    print(f"  gradient_steps: {config.gradient_steps}")
    print(f"  gamma: {config.gamma}")
    print(f"  tau: {config.tau}")
    print(f"  use_sde: {config.use_sde}")
    print(f"  action_noise_std: {config.action_noise_std}")
    print(f"  ent_coef: {config.ent_coef}")
    print(f"  target_entropy: {config.target_entropy}")
    print(f"  min_ent_coef: {config.min_ent_coef}")
    print(f"  dense_reward_shaping: {getattr(config, 'dense_reward_shaping', False)}")
    print(f"  dense_reward_style: {getattr(config, 'dense_reward_style', 'standard')}")
    print(f"  task_geometry_features: {getattr(config, 'task_geometry_features', False)}")
    print(f"  task_stage_features: {getattr(config, 'task_stage_features', False)}")
    print(f"  task_progress_features: {getattr(config, 'task_progress_features', False)}")
    print(f"  residual_guidance: {getattr(config, 'residual_guidance', False)}")
    print(f"  residual_action_scale: {getattr(config, 'residual_action_scale', 0.1)}")
    print(f"  expert_demo_episodes: {getattr(config, 'expert_demo_episodes', 0)}")
    print(f"  expert_demo_style: {getattr(config, 'expert_demo_style', 'staged')}")
    print(f"  expert_hint_style: {getattr(config, 'expert_hint_style', 'staged')}")
    print(f"  bc_pretrain_epochs: {getattr(config, 'bc_pretrain_epochs', 0)}")
    print(f"  demo_prefill_passes: {getattr(config, 'demo_prefill_passes', 1)}")
    print(f"  safe_curriculum: {config.safe_curriculum}")
    print("  env terminate_on_success: False (fixed horizon with TimeLimit)")


def configure_runtime(config: SACConfig) -> None:
    """Torch/CUDA 运行时调优"""
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

    print("⚙️ 运行时配置")
    print(f"  device: {config.device}")
    print(f"  torch_num_threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"  cuda available: True ({torch.cuda.device_count()} GPU)")
        print(f"  TF32 enabled: {bool(config.enable_tf32)}")
    else:
        print("  cuda available: False")


def is_pick_and_place_sparse(env_name: str) -> bool:
    return ("PickAndPlace" in env_name) and ("Sparse" in env_name)


def create_env(
    env_name,
    render_mode=None,
    reward_scale=1.0,
    seed: Optional[int] = None,
    dense_reward_shaping: bool = False,
    dense_reward_style: str = "standard",
    task_geometry_features: bool = False,
    task_stage_features: bool = False,
    task_progress_features: bool = False,
    expert_hint_style: str = "staged",
    residual_guidance: bool = False,
    residual_action_scale: float = 0.1,
    safe_curriculum: bool = False,
    safe_curriculum_total_env_steps: Optional[int] = None,
):
    env = gym.make(env_name, render_mode=render_mode)

    if safe_curriculum and is_pick_and_place_sparse(env_name):
        env = SafePickAndPlaceCurriculumWrapper(
            env,
            total_env_steps_target=safe_curriculum_total_env_steps or 1_000_000,
        )

    if task_progress_features and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceTaskProgressWrapper(env, hint_style=expert_hint_style)
    elif task_stage_features and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceStageFeatureWrapper(env, hint_style=expert_hint_style)
    elif task_geometry_features and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceGeometryWrapper(env)
    if residual_guidance and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceResidualGuidanceWrapper(env, residual_scale=residual_action_scale)

    if dense_reward_shaping and is_pick_and_place_sparse(env_name):
        env = PickAndPlaceDenseRewardWrapper(env, reward_style=dense_reward_style)

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
    reward_scale=1.0,
    seed=None,
    start_method="forkserver",
    dense_reward_shaping: bool = False,
    dense_reward_style: str = "standard",
    task_geometry_features: bool = False,
    task_stage_features: bool = False,
    task_progress_features: bool = False,
    expert_hint_style: str = "staged",
    residual_guidance: bool = False,
    residual_action_scale: float = 0.1,
    safe_curriculum: bool = False,
    safe_curriculum_total_env_steps: Optional[int] = None,
):
    def make_env(rank):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_env(
                env_name,
                reward_scale=reward_scale,
                seed=env_seed,
                dense_reward_shaping=dense_reward_shaping,
                dense_reward_style=dense_reward_style,
                task_geometry_features=task_geometry_features,
                task_stage_features=task_stage_features,
                task_progress_features=task_progress_features,
                expert_hint_style=expert_hint_style,
                residual_guidance=residual_guidance,
                residual_action_scale=residual_action_scale,
                safe_curriculum=safe_curriculum,
                safe_curriculum_total_env_steps=safe_curriculum_total_env_steps,
            )

        return _init

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        env_fns = [make_env(i) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)

    if normalize:
        norm_obs_keys = None
        obs_space = vec_env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            goal_keys = {"observation", "achieved_goal", "desired_goal"}
            if goal_keys.issubset(set(obs_space.spaces.keys())):
                # For HER + GoalEnv, keep goals in raw coordinates to avoid
                # reward relabeling mismatch inside replay buffer.
                norm_obs_keys = ["observation"]
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=False,
            norm_obs_keys=norm_obs_keys,
        )
    return vec_env


def create_sac_model(env, config: SACConfig):
    n_actions = env.action_space.shape[-1]
    action_noise = None
    if not config.use_sde:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.action_noise_std * np.ones(n_actions),
        )

    replay_buffer_class = None
    replay_buffer_kwargs = None
    if config.her:
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = dict(
            n_sampled_goal=config.n_sampled_goal,
            goal_selection_strategy=config.goal_selection_strategy,
            copy_info_dict=config.copy_info_dict,
        )

    model = SAC(
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
        ent_coef=config.ent_coef,
        target_entropy=config.target_entropy,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        tensorboard_log=config.log_dir,
        device=config.device,
    )
    return model


def initialize_sac_model(env, config: SACConfig):
    model = create_sac_model(env, config)
    if not config.init_model_path:
        return model

    print(f"🧬 加载初始权重: {config.init_model_path}")
    loaded_model = SAC.load(config.init_model_path, env=env, device=config.device)
    model.set_parameters(loaded_model.get_parameters(), exact_match=False)
    return model


def _print_expert_dataset_stats(dataset: ExpertDataset) -> None:
    print("🧑‍🏫 Expert demo 收集完成")
    print(f"  episodes: {len(dataset.episode_successes)}")
    print(f"  transitions: {dataset.num_transitions}")
    print(f"  episode success rate: {dataset.success_rate:.3f}")


def collect_pickplace_demos_for_training(config: SACConfig) -> Optional[ExpertDataset]:
    if int(getattr(config, "expert_demo_episodes", 0)) <= 0:
        return None
    if not is_pick_and_place_sparse(config.env_name):
        return None

    print("🧪 正在收集 scripted expert trajectory...")

    def _env_fn():
        return create_env(
            config.env_name,
            reward_scale=config.reward_scale,
            dense_reward_shaping=getattr(config, "dense_reward_shaping", False),
            dense_reward_style=getattr(config, "dense_reward_style", "standard"),
            task_geometry_features=getattr(config, "task_geometry_features", False),
            task_stage_features=getattr(config, "task_stage_features", False),
            task_progress_features=getattr(config, "task_progress_features", False),
            expert_hint_style=getattr(config, "expert_hint_style", "staged"),
            residual_guidance=False,
            residual_action_scale=getattr(config, "residual_action_scale", 0.1),
            safe_curriculum=False,
        )

    dataset = collect_pickplace_expert_dataset(
        _env_fn,
        num_episodes=int(config.expert_demo_episodes),
        seed_start=int(config.seed or 0),
        style=getattr(config, "expert_demo_style", "staged"),
        residual_guidance=getattr(config, "residual_guidance", False),
        residual_scale=getattr(config, "residual_action_scale", 0.1),
    )
    _print_expert_dataset_stats(dataset)
    return dataset


def run_behavior_cloning_pretrain(model, dataset: ExpertDataset, config: SACConfig) -> None:
    epochs = int(getattr(config, "bc_pretrain_epochs", 0))
    if dataset is None or epochs <= 0 or dataset.num_transitions == 0:
        return

    batch_size = max(1, int(getattr(config, "bc_batch_size", 512)))
    learning_rate = getattr(config, "bc_learning_rate", None) or config.learning_rate
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=float(learning_rate))

    model.policy.set_training_mode(True)
    indices = np.arange(dataset.num_transitions)

    print("🎓 开始 Actor behavior cloning 预训练")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {learning_rate}")

    for epoch in range(epochs):
        np.random.shuffle(indices)
        epoch_losses = []
        for start in range(0, dataset.num_transitions, batch_size):
            batch_indices = indices[start:start + batch_size]
            obs_batch = {
                key: dataset.observations[key][batch_indices]
                for key in dataset.observations
            }
            obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
            mean_actions, log_std, kwargs = model.actor.get_action_dist_params(obs_tensor)
            distribution = model.actor.action_dist.proba_distribution(mean_actions, log_std)
            target_actions = torch.as_tensor(
                dataset.actions[batch_indices],
                device=mean_actions.device,
            )
            target_actions = torch.clamp(target_actions, -0.999, 0.999)
            predicted_actions = torch.tanh(mean_actions)
            predicted_motion = predicted_actions[:, :-1]
            predicted_gripper = predicted_actions[:, -1:]
            target_motion = target_actions[:, :-1]
            target_gripper = target_actions[:, -1:]
            bc_motion_mse = torch.nn.functional.mse_loss(predicted_motion, target_motion)
            bc_gripper_mse = torch.nn.functional.mse_loss(predicted_gripper, target_gripper)
            bc_nll = -distribution.log_prob(target_actions).mean()
            loss = bc_motion_mse + 4.0 * bc_gripper_mse + 0.05 * bc_nll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        print(f"  epoch {epoch + 1:02d}/{epochs}: loss={np.mean(epoch_losses):.6f}")

    model.policy.set_training_mode(False)


def _replay_buffer_batch_indices(total: int, batch_size: int):
    end = total - (total % batch_size)
    for start in range(0, end, batch_size):
        yield slice(start, start + batch_size)


def prefill_replay_buffer_from_dataset(model, dataset: ExpertDataset) -> None:
    if dataset is None or dataset.num_transitions == 0:
        return
    if not getattr(model, "replay_buffer", None):
        return

    buffer_envs = int(getattr(model, "n_envs", 1))
    repeat_passes = max(1, int(getattr(model, "_demo_prefill_passes", 1)))
    print(f"📚 开始用 expert demo 预填充 replay buffer (n_envs={buffer_envs}, passes={repeat_passes})")

    added_transitions = 0
    for _ in range(repeat_passes):
        for batch_slice in _replay_buffer_batch_indices(dataset.num_transitions, buffer_envs):
            obs_batch = {
                key: dataset.observations[key][batch_slice]
                for key in dataset.observations
            }
            next_obs_batch = {
                key: dataset.next_observations[key][batch_slice]
                for key in dataset.next_observations
            }
            action_batch = dataset.actions[batch_slice]
            reward_batch = dataset.rewards[batch_slice]
            done_batch = dataset.dones[batch_slice]
            info_batch = dataset.infos[batch_slice]
            model.replay_buffer.add(
                obs_batch,
                next_obs_batch,
                action_batch,
                reward_batch,
                done_batch,
                info_batch,
            )
            added_transitions += int(action_batch.shape[0])

    print(f"  added transitions: {added_transitions}")


def evaluate_manual_any_success(model, config: SACConfig, episodes: int = 10, seed_start: int = 10_000) -> float:
    successes = 0
    for episode_idx in range(max(int(episodes), 1)):
        env = create_env(
            config.env_name,
            reward_scale=config.reward_scale,
            dense_reward_shaping=False,
            task_geometry_features=getattr(config, "task_geometry_features", False),
            task_stage_features=getattr(config, "task_stage_features", False),
            task_progress_features=getattr(config, "task_progress_features", False),
            residual_guidance=getattr(config, "residual_guidance", False),
            residual_action_scale=getattr(config, "residual_action_scale", 0.1),
            safe_curriculum=False,
        )
        observation, _ = env.reset(seed=seed_start + episode_idx)
        any_success = False
        for _ in range(200):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = env.step(action)
            any_success = any_success or bool(info.get("is_success", False))
            if terminated or truncated:
                break
        successes += int(any_success)
        env.close()
    return successes / max(int(episodes), 1)


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
        "final_any_success_rate": float(getattr(training_callback, "recent_any_success_rate", 0.0)),
        "total_episodes": int(training_callback.episode_count),
        "config": serializable_config,
    }


def _save_summary(summary, *paths):
    for summary_path in paths:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"📄 已保存训练摘要: {summary_path}")


def train_sac(config: SACConfig):
    configure_runtime(config)

    print("🚀 开始 SAC 训练！")
    print(f"🎯 环境: {config.env_name}")
    print(f"📊 总训练步数: {config.total_timesteps:,}")
    print(f"⚙️ n_envs: {config.n_envs}")
    print(f"⚙️ start_method: {config.vec_env_start_method}")
    print(f"⚙️ HER: {config.her}")
    print("⚙️ Curriculum: removed (always full task)")
    print(f"📁 结果保存位置: {config.exp_dir}")
    print("-" * 60)

    config.create_directories()

    print("🏗️ 正在创建环境...")
    env = create_vec_env(
        config.env_name,
        n_envs=config.n_envs,
        normalize=config.normalize_env,
        reward_scale=config.reward_scale,
        seed=config.seed,
        start_method=config.vec_env_start_method,
        dense_reward_shaping=getattr(config, "dense_reward_shaping", False),
        dense_reward_style=getattr(config, "dense_reward_style", "standard"),
        task_geometry_features=getattr(config, "task_geometry_features", False),
        task_stage_features=getattr(config, "task_stage_features", False),
        task_progress_features=getattr(config, "task_progress_features", False),
        expert_hint_style=getattr(config, "expert_hint_style", "staged"),
        residual_guidance=getattr(config, "residual_guidance", False),
        residual_action_scale=getattr(config, "residual_action_scale", 0.1),
        safe_curriculum=config.safe_curriculum,
        safe_curriculum_total_env_steps=max(config.total_timesteps // max(config.n_envs, 1), 1),
    )

    eval_env = None
    if config.enable_eval_callback or config.n_eval_episodes > 0:
        eval_env = create_vec_env(
            config.env_name,
            n_envs=1,
            normalize=config.normalize_env,
            reward_scale=config.reward_scale,
            seed=config.seed,
            start_method=config.vec_env_start_method,
            dense_reward_shaping=False,
            dense_reward_style=getattr(config, "dense_reward_style", "standard"),
            task_geometry_features=getattr(config, "task_geometry_features", False),
            task_stage_features=getattr(config, "task_stage_features", False),
            task_progress_features=getattr(config, "task_progress_features", False),
            expert_hint_style=getattr(config, "expert_hint_style", "staged"),
            residual_guidance=getattr(config, "residual_guidance", False),
            residual_action_scale=getattr(config, "residual_action_scale", 0.1),
            safe_curriculum=False,
        )
        if isinstance(eval_env, VecNormalize):
            eval_env.training = False
            eval_env.norm_reward = False

    logger_path = os.path.join(config.log_dir, "tensorboard")
    new_logger = configure(logger_path, ["stdout", "csv", "tensorboard"])

    print("\n🧠 正在初始化 SAC 模型...")
    model = initialize_sac_model(env, config)
    model.set_logger(new_logger)
    model._demo_prefill_passes = int(getattr(config, "demo_prefill_passes", 1))

    expert_dataset = collect_pickplace_demos_for_training(config)
    if expert_dataset is not None:
        run_behavior_cloning_pretrain(model, expert_dataset, config)
        if getattr(config, "prefill_replay_buffer", True):
            prefill_replay_buffer_from_dataset(model, expert_dataset)
        warmstart_success = evaluate_manual_any_success(
            model,
            config,
            episodes=10,
            seed_start=int(config.seed or 0) + 10_000,
        )
        print(f"🔍 BC warm-start any-success@10ep: {warmstart_success:.3f}")

    eval_freq = _to_callback_frequency(config.eval_freq, config.n_envs)
    checkpoint_freq = _to_callback_frequency(config.checkpoint_freq, config.n_envs)

    print(
        f"📈 评估周期: {config.eval_freq} env-steps "
        f"(callback step {eval_freq}, n_envs={config.n_envs})"
    )
    print(
        f"💾 检查点周期: {config.checkpoint_freq} env-steps "
        f"(callback step {checkpoint_freq}, n_envs={config.n_envs})"
    )

    callbacks = []
    training_callback = TrainingCallback(config)
    callbacks.append(training_callback)

    if config.enable_eval_callback and eval_env is not None and config.n_eval_episodes > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=config.model_dir,
            log_path=os.path.join(config.log_dir, "eval"),
            eval_freq=eval_freq,
            n_eval_episodes=config.n_eval_episodes,
            deterministic=config.eval_deterministic,
            verbose=1,
        )
        callbacks.append(eval_callback)

    if config.enable_checkpoint_callback:
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=config.checkpoint_dir,
            name_prefix=f"sac_{config.env_name}",
            save_replay_buffer=config.save_replay_buffer_checkpoints,
            save_vecnormalize=config.save_vecnormalize_checkpoints,
        )
        callbacks.append(checkpoint_callback)

    print("\n🎯 开始训练！")
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
        print("\n⏸️ 训练已中断。")

    end_time = time.time()
    training_time = end_time - start_time

    print("\n✅ 训练完成！")
    print(f"⏱️ 总训练时间: {training_time / 3600:.2f}小时")

    final_model_path = os.path.join(config.model_dir, "final_model")
    model.save(final_model_path)
    if hasattr(env, "save"):
        env.save(os.path.join(config.model_dir, "vec_normalize.pkl"))
    print(f"💾 最终模型已保存: {final_model_path}")

    if eval_env is None:
        eval_env = create_vec_env(
            config.env_name,
            n_envs=1,
            normalize=config.normalize_env,
            reward_scale=config.reward_scale,
            seed=config.seed,
            start_method=config.vec_env_start_method,
            dense_reward_shaping=False,
            dense_reward_style=getattr(config, "dense_reward_style", "standard"),
            task_geometry_features=getattr(config, "task_geometry_features", False),
            task_stage_features=getattr(config, "task_stage_features", False),
            task_progress_features=getattr(config, "task_progress_features", False),
            expert_hint_style=getattr(config, "expert_hint_style", "staged"),
            residual_guidance=getattr(config, "residual_guidance", False),
            residual_action_scale=getattr(config, "residual_action_scale", 0.1),
            safe_curriculum=False,
        )
        if isinstance(eval_env, VecNormalize):
            eval_env.training = False
            eval_env.norm_reward = False

    if isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
        sync_envs_normalization(env, eval_env)

    print("\n📊 最终评估...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )
    print(f"📈 最终评估结果: {mean_reward:.2f} ± {std_reward:.2f}")

    summary = _make_summary(config, training_time, mean_reward, std_reward, training_callback)
    _save_summary(
        summary,
        os.path.join(config.exp_dir, "training_summary.json"),
        os.path.join(config.log_dir, "training_summary.json"),
    )
    return model, env


def apply_pickplace_sparse_defaults(config: SACConfig, args) -> SACConfig:
    if not is_pick_and_place_sparse(config.env_name):
        return config

    profile = getattr(args, "pickplace_profile", "auto")
    if profile == "auto":
        profile = "strong"
    config.pickplace_profile = profile
    explicit_demo_episodes = int(getattr(args, "expert_demo_episodes", 0))
    use_expert_demos = explicit_demo_episodes > 0
    if profile == "strong" and explicit_demo_episodes <= 0:
        use_expert_demos = True

    if args.reward_scale is None:
        config.reward_scale = 1.0
    if args.her is None:
        config.her = False if use_expert_demos else True
    config.curriculum = False
    if args.safe_curriculum is None:
        config.safe_curriculum = profile == "strong"
    else:
        config.safe_curriculum = bool(args.safe_curriculum)
    if args.n_envs is None:
        config.n_envs = 1

    if profile == "legacy":
        if args.tau is None:
            config.tau = 0.005
        if args.learning_rate is None:
            config.learning_rate = 3e-4
        if args.batch_size is None:
            config.batch_size = 512
        if args.learning_starts is None:
            config.learning_starts = 10_000
        if args.gradient_steps is None:
            config.gradient_steps = max(2, int(config.n_envs))
        if args.gamma is None:
            config.gamma = 0.98
        if args.use_sde is None:
            config.use_sde = False
        if args.action_noise_std is None:
            config.action_noise_std = 0.30
        if args.ent_coef is None:
            config.ent_coef = "auto_0.2"
        if args.target_entropy is None:
            config.target_entropy = "auto"
        return config

    if profile == "direct":
        if args.tau is None:
            config.tau = 0.005
        if args.learning_rate is None:
            config.learning_rate = 3e-4
        if args.batch_size is None:
            config.batch_size = 512
        if args.learning_starts is None:
            config.learning_starts = 5_000
        if args.gradient_steps is None:
            config.gradient_steps = max(4, int(config.n_envs))
        if args.gamma is None:
            config.gamma = 0.98
        if args.use_sde is None:
            config.use_sde = False
        if args.action_noise_std is None:
            config.action_noise_std = 0.20
        if args.ent_coef is None:
            config.ent_coef = "auto_0.5"
        if args.target_entropy is None:
            config.target_entropy = -1.0
        if getattr(args, "min_ent_coef", None) is None:
            config.min_ent_coef = 0.01
        if getattr(args, "dense_reward_shaping", None) is None:
            config.dense_reward_shaping = True
        if getattr(args, "dense_reward_style", None) is None:
            config.dense_reward_style = "direct"
        if getattr(args, "task_geometry_features", None) is None:
            config.task_geometry_features = True
        if getattr(args, "task_stage_features", None) is None:
            config.task_stage_features = False
        if getattr(args, "task_progress_features", None) is None:
            config.task_progress_features = False
        if getattr(args, "residual_guidance", None) is None:
            config.residual_guidance = False
        if args.normalize_env is None:
            config.normalize_env = False
        if args.safe_curriculum is None:
            config.safe_curriculum = False
        if explicit_demo_episodes <= 0:
            config.expert_demo_episodes = 0
            config.bc_pretrain_epochs = 0
            config.prefill_replay_buffer = False
            config.demo_prefill_passes = 1
        elif getattr(args, "expert_demo_style", None) is None:
            config.expert_demo_style = "direct"
        if config.dense_reward_shaping:
            config.copy_info_dict = True
        if args.n_sampled_goal is None:
            config.n_sampled_goal = 8
        if args.goal_selection_strategy is None:
            config.goal_selection_strategy = "future"
        return config

    # strong profile (default for pick-and-place sparse)
    if args.tau is None:
        config.tau = 0.005
    if args.learning_rate is None:
        config.learning_rate = 1e-4 if use_expert_demos else 3e-4
    if args.batch_size is None:
        config.batch_size = 512
    if args.learning_starts is None:
        config.learning_starts = 0 if use_expert_demos else 20_000
    if args.gradient_steps is None:
        # Keep enough critic updates for sparse HER.
        config.gradient_steps = max(4, int(config.n_envs))
    if args.gamma is None:
        config.gamma = 0.98
    if args.use_sde is None:
        config.use_sde = False
    if args.action_noise_std is None:
        config.action_noise_std = 0.10 if use_expert_demos else 0.20
    if args.ent_coef is None:
        config.ent_coef = "auto_0.2" if use_expert_demos else "auto_1.0"
    if args.target_entropy is None:
        # Keep higher exploration than SAC default (auto=-action_dim).
        config.target_entropy = -1.0
    if getattr(args, "min_ent_coef", None) is None:
        config.min_ent_coef = 0.005 if use_expert_demos else 0.03
    if getattr(args, "dense_reward_shaping", None) is None:
        config.dense_reward_shaping = True
    if getattr(args, "dense_reward_style", None) is None:
        config.dense_reward_style = "standard"
    if getattr(args, "task_progress_features", None) is None:
        config.task_progress_features = True
    if getattr(args, "residual_guidance", None) is None:
        config.residual_guidance = True
    if args.normalize_env is None:
        config.normalize_env = False
    if args.safe_curriculum is None:
        config.safe_curriculum = False
    if config.dense_reward_shaping:
        config.copy_info_dict = True
    if use_expert_demos and args.expert_demo_episodes <= 0:
        config.expert_demo_episodes = 64
    if use_expert_demos and config.bc_learning_rate is None:
        config.bc_learning_rate = 3e-4
    if use_expert_demos and getattr(args, "demo_prefill_passes", 1) <= 1:
        config.demo_prefill_passes = 8
    if args.n_sampled_goal is None:
        config.n_sampled_goal = 8
    if args.goal_selection_strategy is None:
        config.goal_selection_strategy = "future"

    if config.n_envs > 8:
        print(
            "[WARN] FrankaPickAndPlaceSparse-v0 usually trains more reliably with "
            "small n_envs (1-8). Current n_envs="
            f"{config.n_envs}."
        )
    return config


def main():
    parser = argparse.ArgumentParser(description="SAC 训练脚本（以 HER 为核心的 pick-and-place sparse 默认配置，移除 curriculum 版）")
    parser.add_argument("--env", type=str, default="FrankaSlideDense-v0", help="环境名称")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="总训练步数")
    parser.add_argument("--exp-name", type=str, default=None, help="实验名称")
    parser.add_argument("--reward-scale", type=float, default=None, help="奖励缩放（未指定时使用各环境默认值）")
    parser.add_argument("--n-envs", type=int, default=None, help="并行环境数量")
    parser.set_defaults(normalize_env=None)
    parser.add_argument("--normalize-env", dest="normalize_env", action="store_true", help="启用 VecNormalize observation normalization")
    parser.add_argument("--no-normalize-env", dest="normalize_env", action="store_false", help="禁用 VecNormalize")
    parser.set_defaults(task_progress_features=None)
    parser.set_defaults(task_geometry_features=None)
    parser.set_defaults(task_stage_features=None)
    parser.add_argument("--task-geometry-features", dest="task_geometry_features", action="store_true", help="为 pick-and-place 添加纯几何特征")
    parser.add_argument("--no-task-geometry-features", dest="task_geometry_features", action="store_false", help="禁用 pick-and-place 纯几何特征")
    parser.add_argument("--task-stage-features", dest="task_stage_features", action="store_true", help="为 pick-and-place 添加 phase one-hot + 几何特征")
    parser.add_argument("--no-task-stage-features", dest="task_stage_features", action="store_false", help="禁用 pick-and-place phase one-hot + 几何特征")
    parser.add_argument("--task-progress-features", dest="task_progress_features", action="store_true", help="为 pick-and-place 添加任务进度特征")
    parser.add_argument("--no-task-progress-features", dest="task_progress_features", action="store_false", help="禁用 pick-and-place 任务进度特征")
    parser.set_defaults(residual_guidance=None)
    parser.add_argument("--residual-guidance", dest="residual_guidance", action="store_true", help="学习叠加在 expert hint 之上的 residual action")
    parser.add_argument("--no-residual-guidance", dest="residual_guidance", action="store_false", help="禁用 residual guidance")
    parser.add_argument("--residual-action-scale", type=float, default=0.1, help="residual guidance action scale")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--device", type=str, default="auto", help="训练设备（auto/cpu/cuda/cuda:0）")
    parser.add_argument("--torch-threads", type=int, default=8, help="torch intra-op threads")
    parser.add_argument("--torch-interop-threads", type=int, default=2, help="torch inter-op threads")
    parser.add_argument("--vec-start-method", type=str, default="forkserver", choices=["fork", "forkserver", "spawn"], help="SubprocVecEnv start method")
    parser.add_argument("--eval-freq", type=int, default=20_000, help="训练中的评估周期（env-steps）")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="训练中的评估回合数")
    parser.add_argument("--checkpoint-freq", type=int, default=200_000, help="检查点保存周期（env-steps）")
    parser.add_argument("--save-replay-buffer", action="store_true", help="在检查点中保存 replay buffer")
    parser.add_argument("--no-eval", action="store_true", help="禁用训练中的评估")
    parser.add_argument("--progress-bar", action="store_true", help="显示 progress bar")
    parser.add_argument("--init-model-path", type=str, default=None, help="用作初始权重的现有 SAC 模型路径")
    parser.add_argument("--policy-width", type=int, default=256, help="策略/价值网络 hidden width")
    parser.add_argument("--policy-depth", type=int, default=2, help="策略/价值网络 hidden depth")
    parser.add_argument("--batch-size", type=int, default=None, help="SAC batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="learning rate")
    parser.add_argument("--learning-starts", type=int, default=None, help="random exploration steps before learning")
    parser.add_argument("--gradient-steps", type=int, default=None, help="gradient steps per rollout step")
    parser.add_argument("--gamma", type=float, default=None, help="discount factor")
    parser.add_argument("--tau", type=float, default=None, help="target network soft update coefficient")
    parser.add_argument("--action-noise-std", type=float, default=None, help="normal action noise std when gSDE is disabled")
    parser.add_argument("--ent-coef", type=str, default=None, help="entropy coefficient, e.g. auto or auto_0.2")
    parser.add_argument("--target-entropy", type=str, default=None, help="target entropy, e.g. auto or -2.0")
    parser.add_argument("--min-ent-coef", type=float, default=None, help="auto entropy coefficient lower bound")
    parser.add_argument("--expert-demo-episodes", type=int, default=0, help="scripted expert demo 的回合数")
    parser.add_argument(
        "--expert-demo-style",
        type=str,
        default=None,
        choices=["staged", "direct"],
        help="expert demo trajectory 风格",
    )
    parser.add_argument(
        "--expert-hint-style",
        type=str,
        default=None,
        choices=["staged", "direct"],
        help="用于 task-progress / residual guidance 的 expert hint 风格",
    )
    parser.add_argument("--bc-epochs", type=int, default=0, help="使用 expert demo 进行 actor behavior cloning 预训练的 epoch 数")
    parser.add_argument("--bc-batch-size", type=int, default=512, help="behavior cloning batch size")
    parser.add_argument("--bc-learning-rate", type=float, default=None, help="behavior cloning learning rate")
    parser.add_argument("--no-demo-prefill", action="store_true", help="禁用 expert demo replay buffer 预填充")
    parser.add_argument("--demo-prefill-passes", type=int, default=1, help="expert demo replay buffer 的重复装载次数")
    parser.set_defaults(dense_reward_shaping=None)
    parser.add_argument("--dense-reward-shaping", dest="dense_reward_shaping", action="store_true", help="在 sparse pick-and-place 训练时使用 dense shaping reward")
    parser.add_argument("--no-dense-reward-shaping", dest="dense_reward_shaping", action="store_false", help="禁用 dense shaping reward")
    parser.add_argument(
        "--dense-reward-style",
        type=str,
        default=None,
        choices=["standard", "direct"],
        help="dense shaping 奖励形式",
    )
    parser.add_argument(
        "--pickplace-profile",
        type=str,
        default="auto",
        choices=["auto", "strong", "legacy", "direct"],
        help="FrankaPickAndPlaceSparse-v0 专用预设（auto 与 strong 相同）",
    )
    parser.add_argument("--no-tf32", action="store_true", help="禁用 Ampere 及以上 GPU 的 TF32")

    parser.set_defaults(her=None, curriculum=None, use_sde=None)
    parser.add_argument("--her", dest="her", action="store_true", help="启用 HER replay buffer")
    parser.add_argument("--no-her", dest="her", action="store_false", help="禁用 HER replay buffer")
    parser.add_argument("--n-sampled-goal", type=int, default=None, help="HER sampled goal 数量（未指定时使用预设/默认值）")
    parser.add_argument("--goal-selection-strategy", type=str, default=None, choices=["future", "final", "episode"], help="HER goal relabeling 策略（未指定时使用预设/默认值）")
    parser.add_argument("--copy-info-dict", action="store_true", help="在 HER reward 重算时复制 info dict")
    parser.set_defaults(safe_curriculum=None)
    parser.add_argument("--safe-curriculum", dest="safe_curriculum", action="store_true", help="在 PickPlace sparse 中保持 reward/termination 不变，仅逐步放宽 reset 分布")
    parser.add_argument("--no-safe-curriculum", dest="safe_curriculum", action="store_false", help="禁用 safe curriculum")

    parser.add_argument("--curriculum", dest="curriculum", action="store_true", help="deprecated: ignored, curriculum has been removed")
    parser.add_argument("--no-curriculum", dest="curriculum", action="store_false", help="deprecated: ignored, curriculum has been removed")

    parser.add_argument("--use-sde", dest="use_sde", action="store_true", help="启用 gSDE")
    parser.add_argument("--no-sde", dest="use_sde", action="store_false", help="禁用 gSDE")

    args = parser.parse_args()

    if args.curriculum:
        print(CURRICULUM_REMOVED_WARNING)

    config = SACConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        experiment_name=args.exp_name,
        normalize_env=True if args.normalize_env is None else args.normalize_env,
        reward_scale=args.reward_scale if args.reward_scale is not None else 0.1,
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
        batch_size=args.batch_size if args.batch_size is not None else 1024,
        learning_rate=args.learning_rate if args.learning_rate is not None else 1e-3,
        learning_starts=args.learning_starts if args.learning_starts is not None else 5_000,
        gradient_steps=args.gradient_steps if args.gradient_steps is not None else 1,
        gamma=args.gamma if args.gamma is not None else 0.95,
        tau=args.tau if args.tau is not None else 0.05,
        action_noise_std=args.action_noise_std if args.action_noise_std is not None else 0.2,
        ent_coef=args.ent_coef if args.ent_coef is not None else "auto",
        target_entropy=args.target_entropy if args.target_entropy is not None else "auto",
        min_ent_coef=args.min_ent_coef,
        use_sde=True if args.use_sde is None else args.use_sde,
        her=False if args.her is None else args.her,
        n_sampled_goal=args.n_sampled_goal if args.n_sampled_goal is not None else 4,
        goal_selection_strategy=args.goal_selection_strategy if args.goal_selection_strategy is not None else "future",
        copy_info_dict=args.copy_info_dict,
        curriculum=False,
        enable_tf32=not args.no_tf32,
        dense_reward_shaping=False if args.dense_reward_shaping is None else args.dense_reward_shaping,
        dense_reward_style=args.dense_reward_style or "standard",
        safe_curriculum=False,
        task_geometry_features=False if args.task_geometry_features is None else args.task_geometry_features,
        task_stage_features=False if args.task_stage_features is None else args.task_stage_features,
        task_progress_features=False if args.task_progress_features is None else args.task_progress_features,
        residual_guidance=False if args.residual_guidance is None else args.residual_guidance,
        residual_action_scale=args.residual_action_scale,
        init_model_path=args.init_model_path,
        expert_demo_episodes=args.expert_demo_episodes,
        expert_demo_style=args.expert_demo_style or "staged",
        expert_hint_style=args.expert_hint_style or "staged",
        bc_pretrain_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        prefill_replay_buffer=not args.no_demo_prefill,
        demo_prefill_passes=args.demo_prefill_passes,
    )

    config = apply_pickplace_sparse_defaults(config, args)
    print_pickplace_sparse_effective_config(config)

    train_sac(config)
    print("\n" + "=" * 60)
    print("✅ 训练完成！")
    print(f"📁 结果保存位置: {config.exp_dir}")
    print("下一步:")
    print(f"python evaluate/evaluate_with_video.py --exp-dir {config.exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
