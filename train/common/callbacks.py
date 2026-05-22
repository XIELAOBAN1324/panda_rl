"""
训练回调类（统计/保存/IO 改进版）
"""

import csv
import os
from collections import deque

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class TrainingCallback(BaseCallback):
    """跟踪训练进度的回调"""

    def __init__(self, config, verbose=0):
        super().__init__(verbose)
        self.config = config
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.any_success_count = 0
        self.episode_count = 0
        self.csv_file = os.path.join(config.log_dir, "training_log.csv")

        self.best_reward = float("-inf")
        self.recent_rewards = deque(maxlen=100)
        self.recent_successes = deque(maxlen=100)
        self.recent_any_successes = deque(maxlen=100)
        self.recent_collisions = deque(maxlen=100)
        self.recent_plane_violations = deque(maxlen=100)
        self.recent_position_errors = deque(maxlen=100)
        self.recent_orientation_alignments = deque(maxlen=100)
        self.recent_inplane_alignments = deque(maxlen=100)
        self.recent_pose_alignments = deque(maxlen=100)
        self.recent_glass_fits = deque(maxlen=100)
        self.recent_success_rate = 0.0
        self.recent_any_success_rate = 0.0
        self.recent_collision_rate = 0.0
        self.recent_plane_violation_rate = 0.0
        self.recent_position_error = 0.0
        self.recent_orientation_alignment = 0.0
        self.recent_inplane_alignment = 0.0
        self.recent_pose_alignment = 0.0
        self.recent_glass_fits_rate = 0.0

        self.saved_stages = set()
        self.stage_timesteps = config.get_stage_timesteps()

        self.csv_buffer = []
        self.csv_flush_every = max(1, int(getattr(config, "csv_flush_every", 100)))
        self.print_every_episodes = max(1, int(getattr(config, "print_every_episodes", 20)))

        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestep", "Episode", "Reward", "Length",
                "Success", "Any_Success", "Success_Rate", "Any_Success_Rate",
                "Collision", "Plane_Violation", "Position_Error",
                "Orientation_Alignment", "Inplane_Alignment", "Pose_Alignment", "Glass_Fits_Window",
                "Collision_Rate", "Plane_Violation_Rate", "Recent_Position_Error",
                "Recent_Orientation_Alignment", "Recent_Inplane_Alignment", "Recent_Pose_Alignment",
                "Glass_Fits_Rate",
                "Best_Reward", "Stage", "Curriculum_Stage", "Curriculum_Progress", "Env_Index"
            ])

    def _on_step(self) -> bool:
        self._clamp_entropy_coefficient()
        self._check_stage_save()

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        if len(dones) > 0:
            for env_idx, done in enumerate(dones):
                if done:
                    info = infos[env_idx] if env_idx < len(infos) else {}
                    self._handle_episode_end(info, env_idx)

        return True

    def _clamp_entropy_coefficient(self) -> None:
        min_ent_coef = getattr(self.config, "min_ent_coef", None)
        if min_ent_coef is None:
            return
        if not hasattr(self.model, "log_ent_coef") or self.model.log_ent_coef is None:
            return

        min_log_ent_coef = float(np.log(float(min_ent_coef)))
        with torch.no_grad():
            self.model.log_ent_coef.data.clamp_(min=min_log_ent_coef)

    def _on_training_end(self) -> None:
        self._flush_csv_buffer()

    def _check_stage_save(self):
        if not getattr(self.config, "save_stage_models", True):
            return

        current_timestep = self.num_timesteps
        vec_normalize_env = self.model.get_vec_normalize_env()

        for stage_name, stage_timestep in self.stage_timesteps.items():
            if stage_name in self.saved_stages or current_timestep < stage_timestep:
                continue

            if stage_name != "0_random":
                model_path = os.path.join(self.config.model_dir, f"stage_{stage_name}.zip")
                self.model.save(model_path)

            if vec_normalize_env is not None:
                vecnorm_path = os.path.join(
                    self.config.model_dir,
                    f"stage_{stage_name}_vecnormalize.pkl",
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
                recent_any_successes=np.array(list(self.recent_any_successes), dtype=int),
            )

            self.saved_stages.add(stage_name)
            print(f"💾 阶段已保存: {stage_name}（Step {current_timestep}）")

    def _handle_episode_end(self, info, env_idx: int):
        if "episode" not in info:
            return

        episode_reward = info["episode"]["r"]
        episode_length = info["episode"]["l"]
        is_success = bool(info.get("is_success", False))
        episode_any_success = bool(info.get("episode_any_success", is_success))
        collision = bool(info.get("collision", False))
        plane_violation = bool(info.get("plane_violation", False))
        position_error = float(info.get("position_error", np.nan))
        orientation_alignment = float(info.get("orientation_alignment", np.nan))
        inplane_alignment = float(info.get("inplane_alignment", np.nan))
        pose_alignment = float(info.get("pose_alignment", np.nan))
        glass_fits_window = bool(info.get("glass_fits_window", False))

        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.recent_rewards.append(episode_reward)
        self.recent_successes.append(int(is_success))
        self.recent_any_successes.append(int(episode_any_success))
        self.recent_collisions.append(int(collision))
        self.recent_plane_violations.append(int(plane_violation))
        if np.isfinite(position_error):
            self.recent_position_errors.append(position_error)
        if np.isfinite(orientation_alignment):
            self.recent_orientation_alignments.append(orientation_alignment)
        if np.isfinite(inplane_alignment):
            self.recent_inplane_alignments.append(inplane_alignment)
        if np.isfinite(pose_alignment):
            self.recent_pose_alignments.append(pose_alignment)
        self.recent_glass_fits.append(int(glass_fits_window))
        self.episode_count += 1

        if is_success:
            self.success_count += 1
        if episode_any_success:
            self.any_success_count += 1

        self.recent_success_rate = float(np.mean(self.recent_successes)) if self.recent_successes else 0.0
        self.recent_any_success_rate = (
            float(np.mean(self.recent_any_successes)) if self.recent_any_successes else 0.0
        )
        self.recent_collision_rate = (
            float(np.mean(self.recent_collisions)) if self.recent_collisions else 0.0
        )
        self.recent_plane_violation_rate = (
            float(np.mean(self.recent_plane_violations)) if self.recent_plane_violations else 0.0
        )
        self.recent_position_error = (
            float(np.mean(self.recent_position_errors)) if self.recent_position_errors else 0.0
        )
        self.recent_orientation_alignment = (
            float(np.mean(self.recent_orientation_alignments)) if self.recent_orientation_alignments else 0.0
        )
        self.recent_inplane_alignment = (
            float(np.mean(self.recent_inplane_alignments)) if self.recent_inplane_alignments else 0.0
        )
        self.recent_pose_alignment = (
            float(np.mean(self.recent_pose_alignments)) if self.recent_pose_alignments else 0.0
        )
        self.recent_glass_fits_rate = (
            float(np.mean(self.recent_glass_fits)) if self.recent_glass_fits else 0.0
        )

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            print(
                f"🏆 新的最高奖励！{episode_reward:.2f} "
                f"（最近成功率: {self.recent_success_rate:.3f}）"
            )

        current_stage = self._get_current_stage()
        if self.episode_count % self.print_every_episodes == 0:
            self._print_progress()

        self.logger.record("rollout/any_success_rate", self.recent_any_success_rate)
        self.logger.record("rollout/collision_rate", self.recent_collision_rate)
        self.logger.record("rollout/plane_violation_rate", self.recent_plane_violation_rate)
        self.logger.record("rollout/position_error", self.recent_position_error)
        self.logger.record("rollout/orientation_alignment", self.recent_orientation_alignment)
        self.logger.record("rollout/inplane_alignment", self.recent_inplane_alignment)
        self.logger.record("rollout/pose_alignment", self.recent_pose_alignment)
        self.logger.record("rollout/glass_fits_rate", self.recent_glass_fits_rate)
        self._save_to_csv(
            episode_reward,
            episode_length,
            is_success,
            episode_any_success,
            current_stage,
            info,
            env_idx,
        )

    def _get_current_stage(self) -> str:
        current_timestep = self.num_timesteps
        current_stage = "0_random"
        for stage_name, stage_timestep in sorted(self.stage_timesteps.items(), key=lambda x: x[1]):
            if current_timestep >= stage_timestep:
                current_stage = stage_name
            else:
                break
        return current_stage

    def _print_progress(self):
        if not self.recent_rewards:
            return

        avg_reward = float(np.mean(self.recent_rewards))
        avg_length = (
            float(np.mean(self.episode_lengths[-50:]))
            if len(self.episode_lengths) >= 50
            else float(np.mean(self.episode_lengths))
        )

        print(
            f"📊 Episode {self.episode_count:5d} | "
            f"Step {self.num_timesteps:9d} | "
            f"Reward: {self.episode_rewards[-1]:8.2f} | "
            f"Avg: {avg_reward:8.2f} | "
            f"Len: {avg_length:6.1f} | "
            f"Success: {self.recent_success_rate:.3f} | "
            f"AnySuccess: {self.recent_any_success_rate:.3f} | "
            f"Collision: {self.recent_collision_rate:.3f} | "
            f"Plane: {self.recent_plane_violation_rate:.3f} | "
            f"PosErr: {self.recent_position_error:.4f} | "
            f"Align: {self.recent_orientation_alignment:.3f} | "
            f"InPlane: {self.recent_inplane_alignment:.3f} | "
            f"PoseAlign: {self.recent_pose_alignment:.3f} | "
            f"Fits: {self.recent_glass_fits_rate:.3f}"
        )

    def _save_to_csv(self, episode_reward, episode_length, is_success, any_success, current_stage, info, env_idx):
        curriculum_stage = info.get("curriculum_stage", "")
        curriculum_progress = info.get("curriculum_progress", "")
        self.csv_buffer.append([
            self.num_timesteps,
            self.episode_count,
            episode_reward,
            episode_length,
            is_success,
            any_success,
            self.recent_success_rate,
            self.recent_any_success_rate,
            bool(info.get("collision", False)),
            bool(info.get("plane_violation", False)),
            float(info.get("position_error", np.nan)),
            float(info.get("orientation_alignment", np.nan)),
            float(info.get("inplane_alignment", np.nan)),
            float(info.get("pose_alignment", np.nan)),
            bool(info.get("glass_fits_window", False)),
            self.recent_collision_rate,
            self.recent_plane_violation_rate,
            self.recent_position_error,
            self.recent_orientation_alignment,
            self.recent_inplane_alignment,
            self.recent_pose_alignment,
            self.recent_glass_fits_rate,
            self.best_reward,
            current_stage,
            curriculum_stage,
            curriculum_progress,
            env_idx,
        ])
        if len(self.csv_buffer) >= self.csv_flush_every:
            self._flush_csv_buffer()

    def _flush_csv_buffer(self):
        if not self.csv_buffer:
            return
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.csv_buffer)
        self.csv_buffer.clear()
