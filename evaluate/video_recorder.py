"""
视频录制类
"""

import os
import cv2
import json
import re
import shutil
import subprocess
from typing import List, Dict, Optional, Any

import numpy as np


def _to_json_safe(obj: Any):
    """将 numpy 类型转换为可 JSON 序列化的 Python 基本类型"""
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
    """按回合分别录制视频的类"""

    def __init__(self, save_dir: str, fps: int = 30):
        self.save_dir = save_dir
        self.fps = fps
        self.current_episode_frames = []
        self.episode_metadata = []
        self.last_error = ""
        os.makedirs(save_dir, exist_ok=True)

    def start_episode_recording(self):
        self.current_episode_frames = []
        self.last_error = ""

    def add_frame(self, frame: np.ndarray):
        if frame is not None:
            self.current_episode_frames.append(frame.copy())

    def end_episode_recording(self, episode_info: Dict) -> Optional[str]:
        if not self.current_episode_frames:
            self.last_error = "no rendered frames were captured"
            print(f"⚠️ 视频保存失败: {self.last_error}")
            return None

        episode_id = int(episode_info.get("episode_id", 0))
        success_str = "SUCCESS" if bool(episode_info.get("success", False)) else "FAIL"
        failure_reason = self._safe_filename_token(str(episode_info.get("primary_failure_reason", "unknown")))
        reward = float(episode_info.get("reward", 0.0))

        reason_suffix = "" if success_str == "SUCCESS" else f"_{failure_reason}"
        filename = f"ep{episode_id:03d}_{success_str}{reason_suffix}_reward{reward:.1f}.mp4"
        video_path = os.path.join(self.save_dir, filename)

        version = 1
        while os.path.exists(video_path) and version < 100:
            filename = f"ep{episode_id:03d}_{success_str}{reason_suffix}_reward{reward:.1f}_v{version}.mp4"
            video_path = os.path.join(self.save_dir, filename)
            version += 1

        success = self._save_video(video_path, self.current_episode_frames)
        if success:
            safe_info = dict(episode_info)
            safe_info["video_path"] = str(video_path)
            safe_info["frame_count"] = int(len(self.current_episode_frames))
            safe_info["fps"] = int(self.fps)
            safe_info["duration"] = float(len(self.current_episode_frames) / self.fps)
            self.episode_metadata.append(_to_json_safe(safe_info))
            return video_path
        print(f"⚠️ 视频保存失败: {self.last_error or 'unknown ffmpeg error'}")
        return None

    def _safe_filename_token(self, value: str) -> str:
        token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
        return token or "unknown"

    def _prepare_frames(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        prepared = []
        first = np.asarray(frames[0])
        height, width = first.shape[:2]
        target_height = max(2, height - (height % 2))
        target_width = max(2, width - (width % 2))

        for frame in frames:
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
                raise ValueError(f"unsupported frame shape: {frame.shape}")
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if frame.shape[:2] != (target_height, target_width):
                frame = frame[:target_height, :target_width]
            prepared.append(np.ascontiguousarray(frame))
        return prepared

    def _ffmpeg_candidates(self) -> List[str]:
        candidates = []
        for key in ("PANDA_RL_FFMPEG", "FFMPEG_BINARY"):
            value = os.environ.get(key)
            if value:
                candidates.append(value)

        path_ffmpeg = shutil.which("ffmpeg")
        if path_ffmpeg:
            candidates.append(path_ffmpeg)

        candidates.extend(["/usr/bin/ffmpeg", "/bin/ffmpeg"])

        seen = set()
        unique = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique.append(candidate)
        return unique

    def _remove_incomplete_video(self, video_path: str) -> None:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except OSError:
            pass

    def _save_video_with_ffmpeg(self, video_path: str, frames: List[np.ndarray]) -> bool:
        height, width = frames[0].shape[:2]
        raw_video = b"".join(frame.tobytes() for frame in frames)
        errors = []

        for ffmpeg in self._ffmpeg_candidates():
            if os.path.sep in ffmpeg and not os.path.exists(ffmpeg):
                continue
            cmd = [
                ffmpeg,
                "-y",
                "-loglevel", "error",
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

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                _, stderr_bytes = proc.communicate(input=raw_video)
            except Exception as exc:
                errors.append(f"{ffmpeg}: {exc}")
                self._remove_incomplete_video(video_path)
                continue

            stderr = stderr_bytes.decode("utf-8", errors="replace").strip() if stderr_bytes else ""
            success = proc.returncode == 0 and os.path.exists(video_path) and os.path.getsize(video_path) > 0
            if success:
                return True

            errors.append(f"{ffmpeg}: {stderr or f'exited with code {proc.returncode}'}")
            self._remove_incomplete_video(video_path)

        self.last_error = "; ".join(errors) if errors else "no ffmpeg executable found"
        return False

    def _save_video_with_opencv(self, video_path: str, frames: List[np.ndarray]) -> bool:
        height, width = frames[0].shape[:2]
        errors = []

        for codec in ("mp4v", "XVID", "MJPG"):
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*codec),
                float(self.fps),
                (int(width), int(height)),
            )
            if not writer.isOpened():
                errors.append(f"OpenCV VideoWriter could not open codec {codec}")
                writer.release()
                self._remove_incomplete_video(video_path)
                continue

            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            success = os.path.exists(video_path) and os.path.getsize(video_path) > 0
            if success:
                return True

            errors.append(f"OpenCV VideoWriter codec {codec} produced no output")
            self._remove_incomplete_video(video_path)

        self.last_error = "; ".join(errors) if errors else "OpenCV VideoWriter failed"
        return False

    def _save_video(self, video_path: str, frames: List[np.ndarray]) -> bool:
        if not frames:
            self.last_error = "no frames to encode"
            return False

        try:
            prepared_frames = self._prepare_frames(frames)
        except Exception as exc:
            self.last_error = str(exc)
            return False

        ffmpeg_error = ""
        if self._save_video_with_ffmpeg(video_path, prepared_frames):
            return True
        ffmpeg_error = self.last_error

        if self._save_video_with_opencv(video_path, prepared_frames):
            self.last_error = ""
            return True

        opencv_error = self.last_error
        self.last_error = f"ffmpeg failed ({ffmpeg_error}); OpenCV fallback failed ({opencv_error})"
        return False

    def save_metadata(self):
        if self.episode_metadata:
            metadata_path = os.path.join(self.save_dir, "episode_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(_to_json_safe(self.episode_metadata), f, indent=4, ensure_ascii=False)
