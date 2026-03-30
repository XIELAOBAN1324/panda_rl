"""
학습 콜백 클래스
"""

import os
import csv
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """학습 진행 상황 추적 콜백"""

    def __init__(self, config, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.config = config
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        self.csv_file = os.path.join(config.log_dir, "training_log.csv")

        # 성능 추적
        self.best_reward = float('-inf')
        self.recent_rewards = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        self.recent_success_rate = 0.0

        # 단계 저장을 위한 변수
        self.saved_stages = set()
        self.stage_timesteps = config.get_stage_timesteps()

        # CSV 파일 초기화
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestep', 'Episode', 'Reward', 'Length',
                'Success', 'Success_Rate', 'Best_Reward', 'Stage', 'Env_Index'
            ])

    def _on_step(self) -> bool:
        # 단계별 모델 저장 체크
        self._check_stage_save()

        # 병렬 환경의 모든 종료 에피소드 처리
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        if len(dones) > 0:
            for env_idx, done in enumerate(dones):
                if done:
                    info = infos[env_idx] if env_idx < len(infos) else {}
                    self._handle_episode_end(info, env_idx)

        return True

    def _check_stage_save(self):
        """단계별 모델/정규화 통계 저장"""
        current_timestep = self.num_timesteps
        vec_normalize_env = self.model.get_vec_normalize_env()

        for stage_name, stage_timestep in self.stage_timesteps.items():
            if stage_name in self.saved_stages or current_timestep < stage_timestep:
                continue

            # 0_random 은 실제 랜덤 정책이므로 모델은 저장하지 않음
            if stage_name != '0_random':
                model_path = os.path.join(self.config.model_dir, f"stage_{stage_name}.zip")
                self.model.save(model_path)

            if vec_normalize_env is not None:
                vecnorm_path = os.path.join(
                    self.config.model_dir,
                    f"stage_{stage_name}_vecnormalize.pkl"
                )
                vec_normalize_env.save(vecnorm_path)

            stats_path = os.path.join(self.config.log_dir, f"stage_{stage_name}_stats.npz")
            np.savez(
                stats_path,
                timestep=current_timestep,
                episode_count=self.episode_count,
                success_rate=self.recent_success_rate,
                best_reward=self.best_reward,
                recent_rewards=np.array(list(self.recent_rewards), dtype=float),
                recent_successes=np.array(list(self.recent_successes), dtype=int),
            )

            self.saved_stages.add(stage_name)
            print(f"💾 단계 저장: {stage_name} (Step {current_timestep})")

    def _handle_episode_end(self, info, env_idx: int):
        """에피소드 종료 처리"""
        if 'episode' not in info:
            return

        episode_reward = info['episode']['r']
        episode_length = info['episode']['l']
        is_success = bool(info.get('is_success', False))

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.recent_rewards.append(episode_reward)
        self.recent_successes.append(int(is_success))
        self.episode_count += 1

        if is_success:
            self.success_count += 1

        # 최근 100개 에피소드 기준 성공률 계산
        if self.recent_successes:
            self.recent_success_rate = float(np.mean(self.recent_successes))
        else:
            self.recent_success_rate = 0.0

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            print(
                f"🏆 새로운 최고 보상! {episode_reward:.2f} "
                f"(최근 성공률: {self.recent_success_rate:.3f})"
            )

        current_stage = self._get_current_stage()

        if self.episode_count % 10 == 0:
            self._print_progress()

        self._save_to_csv(episode_reward, episode_length, is_success, current_stage, env_idx)

    def _get_current_stage(self) -> str:
        """현재 학습 단계 반환"""
        current_timestep = self.num_timesteps
        current_stage = '0_random'

        for stage_name, stage_timestep in sorted(self.stage_timesteps.items(), key=lambda x: x[1]):
            if current_timestep >= stage_timestep:
                current_stage = stage_name
            else:
                break

        return current_stage

    def _print_progress(self):
        """진행 상황 출력"""
        if not self.recent_rewards:
            return

        avg_reward = float(np.mean(self.recent_rewards))
        avg_length = (
            float(np.mean(self.episode_lengths[-50:]))
            if len(self.episode_lengths) >= 50
            else float(np.mean(self.episode_lengths))
        )

        print(
            f"📊 Episode {self.episode_count:4d} | "
            f"Step {self.num_timesteps:7d} | "
            f"Reward: {self.episode_rewards[-1]:7.2f} | "
            f"Avg: {avg_reward:7.2f} | "
            f"Len: {avg_length:6.1f} | "
            f"Success: {self.recent_success_rate:.3f}"
        )

    def _save_to_csv(self, episode_reward, episode_length, is_success, current_stage, env_idx):
        """CSV 파일에 로그 저장"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                self.episode_count,
                episode_reward,
                episode_length,
                is_success,
                self.recent_success_rate,
                self.best_reward,
                current_stage,
                env_idx,
            ])
