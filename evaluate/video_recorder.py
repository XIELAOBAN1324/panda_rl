"""
비디오 녹화 클래스들
"""

import os
import cv2
import json
import shutil
from typing import List, Dict, Optional, Any

import numpy as np


def _to_json_safe(obj: Any):
    """numpy 타입을 JSON 직렬화 가능한 파이썬 기본 타입으로 변환"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    return obj


class EpisodeVideoRecorder:
    """에피소드별 개별 비디오 녹화 클래스"""

    def __init__(self, save_dir: str, fps: int = 30):
        self.save_dir = save_dir
        self.fps = fps
        self.current_episode_frames = []
        self.episode_metadata = []
        os.makedirs(save_dir, exist_ok=True)

    def start_episode_recording(self):
        """새 에피소드 녹화 시작"""
        self.current_episode_frames = []

    def add_frame(self, frame: np.ndarray):
        """현재 에피소드에 프레임 추가"""
        if frame is not None:
            self.current_episode_frames.append(frame.copy())

    def end_episode_recording(self, episode_info: Dict) -> Optional[str]:
        """에피소드 녹화 종료 및 비디오 저장"""
        if not self.current_episode_frames:
            return None

        stage = episode_info.get("stage", "unknown")
        episode_id = int(episode_info.get("episode_id", 0))
        success_str = "SUCCESS" if bool(episode_info.get("success", False)) else "FAIL"
        reward = float(episode_info.get("reward", 0.0))

        filename = f"ep{episode_id:03d}_{success_str}_reward{reward:.1f}.mp4"
        video_path = os.path.join(self.save_dir, filename)

        version = 1
        while os.path.exists(video_path) and version < 100:
            filename = f"ep{episode_id:03d}_{success_str}_reward{reward:.1f}_v{version}.mp4"
            video_path = os.path.join(self.save_dir, filename)
            version += 1

        success = self._save_video(video_path, self.current_episode_frames)
        if success:
            safe_info = dict(episode_info)
            safe_info["video_path"] = str(video_path)
            safe_info["frame_count"] = int(len(self.current_episode_frames))
            safe_info["fps"] = int(self.fps)
            safe_info["duration"] = float(len(self.current_episode_frames) / self.fps)

            safe_info = _to_json_safe(safe_info)
            self.episode_metadata.append(safe_info)

            print(f"   ✅ 비디오 저장: {filename}")
            print(
                f"      프레임: {len(self.current_episode_frames)}, "
                f"성공: {success_str}, 보상: {reward:.2f}"
            )
            return video_path

        return None

    def _save_video(self, video_path: str, frames: List[np.ndarray]) -> bool:
        if not frames:
            return False

        try:
            import subprocess
            import numpy as np

            height, width = frames[0].shape[:2]

            cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(self.fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                video_path,
            ]

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            for frame in frames:
                if frame is None:
                    continue
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                # 这里假设 frame 是 RGB
                proc.stdin.write(frame.tobytes())

            proc.stdin.close()
            proc.wait()

            return proc.returncode == 0 and os.path.exists(video_path) and os.path.getsize(video_path) > 0

        except Exception as e:
            print(f"❌ 保存视频错误: {e}")
            return False

    def save_metadata(self):
        """에피소드 메타데이터를 JSON으로 저장"""
        if self.episode_metadata:
            metadata_path = os.path.join(self.save_dir, "episode_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(_to_json_safe(self.episode_metadata), f, indent=4, ensure_ascii=False)
            print(f"   📝 메타데이터 저장: {metadata_path}")


class StageVideoRecorder:
    """학습 단계별 비디오 녹화 시스템"""

    def __init__(self, base_dir: str, fps: int = 30):
        self.base_dir = base_dir
        self.fps = fps
        self.stage_results = {}
        self.best_episodes = []
        os.makedirs(base_dir, exist_ok=True)

    def record_stage_episodes(self, stage_name: str, model, env, num_episodes: int = 3):
        """특정 단계의 에피소드들 녹화"""
        print(f"\n🎬 [{stage_name}] 비디오 녹화 시작 ({num_episodes}개 에피소드)")

        stage_dir = os.path.join(self.base_dir, stage_name)
        os.makedirs(stage_dir, exist_ok=True)

        recorder = EpisodeVideoRecorder(stage_dir, fps=self.fps)
        episode_results = []
        successful_episodes = []
        failed_episodes = []

        attempts = 0
        max_attempts = num_episodes * 5

        while len(episode_results) < num_episodes and attempts < max_attempts:
            attempts += 1

            result = self._record_single_episode(
                recorder, model, env, len(episode_results), stage_name
            )

            if result:
                if result["success"]:
                    successful_episodes.append(result)
                else:
                    failed_episodes.append(result)

            if len(successful_episodes) >= num_episodes:
                episode_results = successful_episodes[:num_episodes]
                break
            elif attempts >= max_attempts - 1:
                episode_results = (successful_episodes + failed_episodes)[:num_episodes]
                break

        recorder.save_metadata()

        self.stage_results[stage_name] = episode_results

        for result in episode_results:
            if (
                not self.best_episodes
                or float(result["reward"]) > min(float(ep["reward"]) for ep in self.best_episodes)
            ):
                self.best_episodes.append(_to_json_safe(result))
                self.best_episodes.sort(key=lambda x: float(x["reward"]), reverse=True)
                self.best_episodes = self.best_episodes[:10]

        success_count = sum(1 for r in episode_results if r.get("success", False))
        avg_reward = np.mean([float(r["reward"]) for r in episode_results]) if episode_results else 0.0

        print(f"✅ [{stage_name}] 완료!")
        print(f"   녹화된 에피소드: {len(episode_results)}개")
        print(f"   성공률: {success_count}/{len(episode_results)}")
        print(f"   평균 보상: {avg_reward:.2f}")

        return episode_results

    def _record_single_episode(self, recorder, model, env, episode_idx, stage_name):
        """단일 에피소드 녹화"""
        recorder.start_episode_recording()

        obs = env.reset()
        total_reward = 0.0
        step_count = 0
        done = False
        success = False

        for _ in range(3):
            rendered = env.render()
            if rendered is not None:
                img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                recorder.add_frame(img)

        while not done:
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                raw_action = env.action_space.sample()
                action = np.array([raw_action])

            obs, rewards, dones, infos = env.step(action)

            done = bool(dones[0])
            total_reward += float(rewards[0])
            step_count += 1

            rendered = env.render()
            if rendered is not None:
                img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                recorder.add_frame(img)

            if bool(infos[0].get("is_success", False)):
                success = True
                for _ in range(15):
                    rendered = env.render()
                    if rendered is not None:
                        img = rendered[0] if isinstance(rendered, (list, tuple)) else rendered
                        recorder.add_frame(img)
                break

            if step_count > 1000:
                break

        episode_info = {
            "episode_id": int(episode_idx),
            "reward": float(total_reward),
            "length": int(step_count),
            "success": bool(success),
            "stage": str(stage_name),
        }

        video_path = recorder.end_episode_recording(episode_info)
        if video_path:
            episode_info["video_path"] = str(video_path)
            return _to_json_safe(episode_info)

        return None

    def create_highlight_reel(self):
        """최고 보상 에피소드들로 하이라이트 비디오 생성"""
        if not self.best_episodes:
            print("❌ 하이라이트 생성 실패: 에피소드 없음")
            return

        print(f"\n📦 하이라이트 폴더 생성 중... (상위 {len(self.best_episodes)}개 에피소드)")

        highlight_dir = os.path.join(self.base_dir, "highlights")
        os.makedirs(highlight_dir, exist_ok=True)

        for i, episode in enumerate(self.best_episodes):
            if "video_path" in episode and os.path.exists(episode["video_path"]):
                original_path = episode["video_path"]
                highlight_filename = (
                    f"best_{i+1:02d}_reward{float(episode['reward']):.1f}_{episode['stage']}.mp4"
                )
                highlight_path = os.path.join(highlight_dir, highlight_filename)

                shutil.copy2(original_path, highlight_path)
                print(f"   ✨ {highlight_filename}")

        highlight_metadata = {
            "best_episodes": _to_json_safe(self.best_episodes),
            "total_stages_evaluated": int(len(self.stage_results)),
            "stages": [str(x) for x in self.stage_results.keys()],
        }

        metadata_path = os.path.join(highlight_dir, "highlight_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(highlight_metadata, f, indent=4, ensure_ascii=False)

        print(f"✅ 하이라이트 완료! {highlight_dir}")

    def save_evaluation_summary(self):
        """전체 평가 요약 저장"""
        summary = {
            "stages_evaluated": [str(x) for x in self.stage_results.keys()],
            "total_episodes": int(sum(len(episodes) for episodes in self.stage_results.values())),
            "stage_summaries": {},
        }

        for stage, episodes in self.stage_results.items():
            if episodes:
                rewards = [float(ep["reward"]) for ep in episodes]
                successes = [bool(ep["success"]) for ep in episodes]

                summary["stage_summaries"][stage] = {
                    "num_episodes": int(len(episodes)),
                    "mean_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "success_rate": float(sum(successes) / len(successes)),
                    "best_reward": float(max(rewards)),
                    "worst_reward": float(min(rewards)),
                }

        summary_path = os.path.join(self.base_dir, "evaluation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_to_json_safe(summary), f, indent=4, ensure_ascii=False)

        print(f"\n📊 평가 요약 저장: {summary_path}")
        return summary
