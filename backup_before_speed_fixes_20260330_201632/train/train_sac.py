#!/usr/bin/env python3
"""
SAC 학습 스크립트
- 순수하게 학습에만 집중
- 결과는 outputs 폴더에 저장
- 비디오 녹화는 별도 스크립트에서 수행
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 사용자 정의 환경 등록
import panda_mujoco_gym

# Common 모듈 임포트
from train.common.config import SACConfig
from train.common.wrappers import RewardScalingWrapper, SuccessTrackingWrapper
from train.common.callbacks import TrainingCallback

# 한 에피소드 당 step을 기본 50 -> 100으로 wrapping하기 위해
from gymnasium.wrappers import TimeLimit


def create_env(env_name, render_mode=None, reward_scale=1.0):
    """환경 생성 (래퍼 적용)"""
    raw = gym.make(env_name, render_mode=render_mode)
    env = TimeLimit(raw, max_episode_steps=100)
    env = Monitor(env)

    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)

    env = SuccessTrackingWrapper(env)
    return env


# 각 워커에 서로 다른 시드를 할당
def create_vec_env(env_name, n_envs=1, normalize=True, reward_scale=1.0, seed=None):
    def make_env(rank):
        def _init():
            env = create_env(env_name, reward_scale=reward_scale)
            if seed is not None:
                env.reset(seed=seed + rank)
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            return env
        return _init

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env(0)])
    else:
        env_fns = [make_env(i) for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method='fork')

    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    return vec_env


def create_sac_model(env, config):
    """SAC 모델 생성"""
    n_actions = env.action_space.shape[-1]

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=config.action_noise_std * np.ones(n_actions)
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
        device="auto"
    )

    return model


def _to_callback_frequency(target_steps: int, n_envs: int) -> int:
    """SB3 callback 는 env.step() 단위로 호출되므로 병렬 환경 수를 반영해 변환"""
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
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"📋 학습 요약 저장: {summary_path}")


def train_sac(config: SACConfig):
    """SAC 학습 메인 함수"""
    print("🚀 SAC 학습 시작!")
    print(f"🎯 환경: {config.env_name}")
    print(f"📊 총 학습 스텝: {config.total_timesteps:,}")
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
    )
    eval_env = create_vec_env(
        config.env_name,
        n_envs=1,
        normalize=config.normalize_env,
        reward_scale=config.reward_scale,
        seed=config.seed,
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

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=config.checkpoint_dir,
        name_prefix=f"sac_{config.env_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    print("\n🚀 학습 시작!")
    print("=" * 60)
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⏸️  학습이 중단되었습니다.")

    end_time = time.time()
    training_time = end_time - start_time

    print("\n✅ 학습 완료!")
    print(f"⏱️  총 학습 시간: {training_time / 3600:.2f}시간")

    final_model_path = os.path.join(config.model_dir, "final_model")
    model.save(final_model_path)
    if hasattr(env, 'save'):
        env.save(os.path.join(config.model_dir, "vec_normalize.pkl"))
    print(f"💾 최종 모델 저장: {final_model_path}")

    print("\n🔍 최종 평가...")
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=50,
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
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="SAC 학습 스크립트")
    parser.add_argument("--env", type=str, default="FrankaSlideDense-v0", help="환경 이름")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="총 학습 스텝")
    parser.add_argument("--exp-name", type=str, default=None, help="실험 이름")
    parser.add_argument("--reward-scale", type=float, default=0.1, help="보상 스케일")
    parser.add_argument("--n-envs", type=int, default=4, help="병렬 환경 개수")
    parser.add_argument("--seed", type=int, default=None, help="난수 시드 (worker마다 seed+rank 적용)")
    args = parser.parse_args()

    config = SACConfig(
        env_name=args.env,
        total_timesteps=args.timesteps,
        experiment_name=args.exp_name,
        reward_scale=args.reward_scale,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    model, env = train_sac(config)

    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print(f"📁 결과 저장 위치: {config.exp_dir}")
    print("📌 다음 단계:")
    print(f"   python evaluate/evaluate_with_video.py --exp-dir {config.exp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
