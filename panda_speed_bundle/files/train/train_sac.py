#!/usr/bin/env python3
"""
SAC 학습 스크립트 (속도/안정성 개선판)
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, sync_envs_normalization

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import panda_mujoco_gym
from train.common.callbacks import TrainingCallback
from train.common.config import SACConfig, _recommended_n_envs
from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper


def configure_runtime(config: SACConfig) -> None:
    """Torch/CUDA 런타임 튜닝"""
    if config.torch_num_threads is not None:
        torch.set_num_threads(int(config.torch_num_threads))
    if config.torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(config.torch_num_interop_threads))
        except RuntimeError:
            # 이미 설정된 경우 무시
            pass

    if config.enable_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("🛠️  런타임 설정")
    print(f"   device: {config.device}")
    print(f"   torch_num_threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        print(f"   cuda available: True ({torch.cuda.device_count()} GPU)")
        print(f"   TF32 enabled: {bool(config.enable_tf32)}")
    else:
        print("   cuda available: False")


def create_env(env_name, render_mode=None, reward_scale=1.0, seed: Optional[int] = None):
    raw = gym.make(env_name, render_mode=render_mode)
    env = TimeLimit(raw, max_episode_steps=100)
    env = Monitor(env)
    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)
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
):
    def make_env(rank):
        def _init():
            env_seed = None if seed is None else seed + rank
            return create_env(env_name, reward_scale=reward_scale, seed=env_seed)
        return _init

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        env_fns = [make_env(i) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)

    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    return vec_env


def create_sac_model(env, config: SACConfig):
    n_actions = env.action_space.shape[-1]
    action_noise = None
    if not config.use_sde:
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=config.action_noise_std * np.ones(n_actions),
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
        verbose=1,
        tensorboard_log=config.log_dir,
        device=config.device,
    )
    return model


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
        print(f"📋 학습 요약 저장: {summary_path}")


def train_sac(config: SACConfig):
    configure_runtime(config)

    print("🚀 SAC 학습 시작!")
    print(f"🎯 환경: {config.env_name}")
    print(f"📊 총 학습 스텝: {config.total_timesteps:,}")
    print(f"⚙️  n_envs: {config.n_envs}")
    print(f"⚙️  start_method: {config.vec_env_start_method}")
    print(f"💾 결과 저장 위치: {config.exp_dir}")
    print("-" * 60)

    config.create_directories()

    print("🏗️  환경 생성 중...")
    env = create_vec_env(
        config.env_name,
        n_envs=config.n_envs,
        normalize=config.normalize_env,
        reward_scale=config.reward_scale,
        seed=config.seed,
        start_method=config.vec_env_start_method,
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
        )
        if isinstance(eval_env, VecNormalize):
            eval_env.training = False
            eval_env.norm_reward = False

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

    callbacks = []
    training_callback = TrainingCallback(config)
    callbacks.append(training_callback)

    if config.enable_eval_callback and eval_env is not None and config.n_eval_episodes > 0:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(config.model_dir, "best_model"),
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

    print("\n🚀 학습 시작!")
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
        print("\n⏸️  학습이 중단되었습니다.")

    end_time = time.time()
    training_time = end_time - start_time

    print("\n✅ 학습 완료!")
    print(f"⏱️  총 학습 시간: {training_time / 3600:.2f}시간")

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
        )
    if isinstance(eval_env, VecNormalize):
        eval_env.training = False
        eval_env.norm_reward = False
    if isinstance(env, VecNormalize) and isinstance(eval_env, VecNormalize):
        sync_envs_normalization(env, eval_env)

    print("\n🔍 최종 평가...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )
    print(f"🏆 최종 평가 결과: {mean_reward:.2f} ± {std_reward:.2f}")

    summary = _make_summary(config, training_time, mean_reward, std_reward, training_callback)
    _save_summary(
        summary,
        os.path.join(config.exp_dir, "training_summary.json"),
        os.path.join(config.log_dir, "training_summary.json"),
    )

    return model, env


def main():
    parser = argparse.ArgumentParser(description="SAC 학습 스크립트 (속도/안정성 개선판)")
    parser.add_argument("--env", type=str, default="FrankaSlideDense-v0", help="환경 이름")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="총 학습 스텝")
    parser.add_argument("--exp-name", type=str, default=None, help="실험 이름")
    parser.add_argument("--reward-scale", type=float, default=0.1, help="보상 스케일")
    parser.add_argument("--n-envs", type=int, default=None, help="병렬 환경 개수")
    parser.add_argument("--seed", type=int, default=None, help="난수 시드")

    parser.add_argument("--device", type=str, default="auto", help="학습 디바이스 (auto/cpu/cuda/cuda:0)")
    parser.add_argument("--torch-threads", type=int, default=8, help="torch intra-op threads")
    parser.add_argument("--torch-interop-threads", type=int, default=2, help="torch inter-op threads")
    parser.add_argument("--vec-start-method", type=str, default="forkserver", choices=["fork", "forkserver", "spawn"], help="SubprocVecEnv start method")

    parser.add_argument("--eval-freq", type=int, default=20_000, help="학습 중 평가 주기 (env-steps)")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="학습 중 평가 에피소드 수")
    parser.add_argument("--checkpoint-freq", type=int, default=200_000, help="체크포인트 저장 주기 (env-steps)")
    parser.add_argument("--save-replay-buffer", action="store_true", help="체크포인트에 replay buffer 저장")
    parser.add_argument("--no-eval", action="store_true", help="학습 중 평가 비활성화")
    parser.add_argument("--progress-bar", action="store_true", help="progress bar 표시")

    parser.add_argument("--policy-width", type=int, default=256, help="정책/가치망 hidden width")
    parser.add_argument("--policy-depth", type=int, default=2, help="정책/가치망 hidden depth")
    parser.add_argument("--batch-size", type=int, default=1024, help="SAC batch size")
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
    print("다음 단계:")
    print(f"python evaluate/evaluate_with_video.py --exp-dir {config.exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
