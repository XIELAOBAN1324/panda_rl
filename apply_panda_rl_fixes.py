from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Callable


class PatchError(RuntimeError):
    pass


def replace_or_fail(text: str, pattern: str, replacement: str, path: pathlib.Path, flags: int = re.MULTILINE | re.DOTALL) -> str:
    new_text, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise PatchError(f"Failed to patch {path}: pattern not found\nPattern: {pattern}")
    return new_text


def write_text(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def patch_evaluate(repo_root: pathlib.Path) -> None:
    path = repo_root / "evaluate" / "evaluate_with_video.py"
    text = path.read_text(encoding="utf-8")

    text = replace_or_fail(
        text,
        r"def evaluate_model_performance\(model, env, num_episodes: int = 50\) -> Dict:.*?def _get_vecnormalize_path",
        '''def evaluate_model_performance(model, env, num_episodes: int = 50) -> Dict:
    """모델 또는 랜덤 정책 성능 평가.

    evaluate_policy()는 성공률이나 개별 rollout 통계를 직접 제공하지 않으므로,
    여기서는 모델/랜덤 정책 모두 동일한 수동 rollout 경로로 평가한다.
    """
    policy_name = '랜덤 정책' if model is None else '모델'
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

            if done and infos[0].get('is_success', False):
                success_count += 1

        rewards.append(total_reward)
        lengths.append(length)

    results = {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'mean_length': float(np.mean(lengths)),
        'success_rate': success_count / num_episodes,
        'num_episodes': num_episodes,
        'all_rewards': [float(r) for r in rewards],
        'all_lengths': [int(l) for l in lengths],
        'rollout_mean_reward': float(np.mean(rewards)),
        'rollout_mean_length': float(np.mean(lengths)),
    }

    print(f"   평균 보상: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   성공률: {results['success_rate']:.3f}")
    print(f"   평균 에피소드 길이: {results['mean_length']:.1f}")

    return results


def _get_vecnormalize_path''',
        path,
    )

    text = replace_or_fail(
        text,
        r"def _build_stage_model_list\(available_models: List\[str\], stages: List\[str\] = None\) -> List\[str\]:.*?def evaluate_experiment",
        '''def _build_stage_model_list(available_models: List[str], stages: List[str] = None) -> List[str]:
    if stages is None or 'all' in stages:
        selected = [m for m in available_models if m.startswith('stage_')]
        if 'best_model' in available_models:
            selected.append('best_model')
        if 'final_model' in available_models:
            selected.append('final_model')
    else:
        selected = []
        for stage in stages:
            normalized_stage = stage.lower()
            if normalized_stage == 'best' and 'best_model' in available_models:
                selected.append('best_model')
                continue
            if normalized_stage == 'final' and 'final_model' in available_models:
                selected.append('final_model')
                continue

            model_name = stage if stage.startswith('stage_') else f'stage_{stage}'
            if model_name in available_models:
                selected.append(model_name)

    def sort_key(name: str):
        if name == 'best_model':
            return (998, name)
        if name == 'final_model':
            return (999, name)
        if name.startswith('stage_'):
            stage_label = name.replace('stage_', '', 1)
            try:
                stage_num = int(stage_label.split('_', 1)[0])
            except ValueError:
                stage_num = 997
            return (stage_num, name)
        return (997, name)

    return sorted(dict.fromkeys(selected), key=sort_key)


def evaluate_experiment''',
        path,
    )

    write_text(path, text)



def patch_launch(repo_root: pathlib.Path) -> None:
    path = repo_root / "launch_multi_seed.py"
    new_text = '''"""
Launch multiple panda_mujoco_gym training jobs across selected GPUs.

NOTE:
When CUDA_VISIBLE_DEVICES is set to a single physical GPU, PyTorch sees that GPU
as cuda:0 inside the child process. Passing cuda:{physical_id} at the same time
causes device mismatches on GPU ids other than 0.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch multiple panda_mujoco_gym training jobs across GPUs")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU ids")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3], help="Seeds to run")
    parser.add_argument("--env", type=str, default="FrankaSlideDense-v0")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--extra-args", type=str, default="", help="Extra args passed to train/train_sac.py")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("No GPU ids provided")

    repo_root = Path(__file__).resolve().parent
    train_script = repo_root / "train" / "train_sac.py"

    procs = []
    for idx, seed in enumerate(args.seeds):
        gpu = gpu_ids[idx % len(gpu_ids)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        cmd = [
            sys.executable,
            str(train_script),
            "--env",
            args.env,
            "--timesteps",
            str(args.timesteps),
            "--n-envs",
            str(args.n_envs),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--exp-name",
            f"{args.env}_seed{seed}_gpu{gpu}",
        ]
        if args.extra_args:
            cmd.extend(args.extra_args.split())

        log_path = repo_root / f"launch_seed{seed}_gpu{gpu}.log"
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=logf, stderr=subprocess.STDOUT)
            procs.append((seed, gpu, proc.pid, str(log_path)))

        print(f"Launched seed={seed} on gpu={gpu}, pid={proc.pid}, device=cuda:0, log={log_path}")

    print("\nAll jobs launched.")
    for seed, gpu, pid, log_path in procs:
        print(f"  seed={seed} gpu={gpu} pid={pid} log={log_path}")


if __name__ == "__main__":
    main()
'''
    write_text(path, new_text)



def patch_train(repo_root: pathlib.Path) -> None:
    path = repo_root / "train" / "train_sac.py"
    text = path.read_text(encoding="utf-8")

    text = replace_or_fail(
        text,
        r"def create_env\(env_name, render_mode=None, reward_scale=1\.0, seed: Optional\[int\] = None\):.*?return env\n",
        '''def create_env(env_name, render_mode=None, reward_scale=1.0, seed: Optional[int] = None):
    env = gym.make(env_name, render_mode=render_mode)

    # Reward scaling과 Monitor/episode 통계의 기준을 맞추기 위해
    # 보상 스케일링을 먼저 적용한 뒤 Monitor를 감싼다.
    if reward_scale != 1.0:
        env = RewardScalingWrapper(env, scale=reward_scale)

    env = Monitor(env)
    env = SuccessTrackingWrapper(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env
''',
        path,
    )

    text = replace_or_fail(
        text,
        r"vec_env = VecNormalize\(vec_env, norm_obs=True, norm_reward=True\)",
        'vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)',
        path,
        flags=re.MULTILINE,
    )

    text = replace_or_fail(
        text,
        r"best_model_save_path=os\.path\.join\(config\.model_dir, \"best_model\"\),",
        'best_model_save_path=config.model_dir,',
        path,
        flags=re.MULTILINE,
    )

    write_text(path, text)



def patch_panda_env(repo_root: pathlib.Path) -> None:
    path = repo_root / "panda_mujoco_gym" / "envs" / "panda_env.py"
    text = path.read_text(encoding="utf-8")

    text = replace_or_fail(
        text,
        r"def step\(self, action\) -> tuple\[ObsType, SupportsFloat, bool, bool, dict\[str, Any\]\]:.*?def _set_action",
        '''def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._mujoco_step(action)
        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()
        ee_position = obs["observation"][:3]
        achieved_goal = obs["achieved_goal"]

        info = {
            "is_success": self._is_success(achieved_goal, self.goal),
            "ee_object_distance": float(np.linalg.norm(ee_position - achieved_goal)),
            "object_height": float(achieved_goal[2] - self.initial_object_height),
        }
        terminated = bool(info["is_success"])
        truncated = bool(self.compute_truncated(achieved_goal, self.goal, info))
        reward = self.compute_reward(achieved_goal, self.goal, info)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)

        reward = -d

        # Pick-and-place 계열은 목표 거리만으로는 탐색이 매우 어려워서,
        # 도달(reach)과 리프트(lift) 신호를 약하게 추가한다.
        if not self.block_gripper:
            ee_object_distance = float(info.get("ee_object_distance", 0.0))
            object_height = max(float(info.get("object_height", 0.0)), 0.0)

            reward -= 0.25 * ee_object_distance

            if self.goal_z_range > 0.0:
                lift_cap = max(self.goal_z_range, self.distance_threshold)
                reward += 0.5 * min(object_height, lift_cap)
                if object_height > self.distance_threshold:
                    reward += 0.25

        return reward

    def _set_action''',
        path,
    )

    text = replace_or_fail(
        text,
        r"def _mujoco_step\(self, action: Optional\[np\.ndarray\] = None\) -> None:\n\s*for _ in range\(10\):\n\s*self\._mujoco\.mj_step\(self\.model, self\.data, nstep=self\.n_substeps\)",
        '''def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)''',
        path,
    )

    text = replace_or_fail(
        text,
        r"if not self\.block_gripper and self\.goal_z_range > 0\.0:\n\s*if self\.np_random\.random\(\) < 0\.3:\n\s*noise\[2\] = 0\.0",
        '''if not self.block_gripper and self.goal_z_range > 0.0:
            # 대부분은 탁자 위 목표를 유지하고, 일부만 공중 목표로 둔다.
            # pick-and-place 초기 학습 난이도를 과도하게 높이지 않기 위함이다.
            if self.np_random.random() < 0.7:
                noise[2] = 0.0''',
        path,
    )

    write_text(path, text)


PATCHERS: list[Callable[[pathlib.Path], None]] = [
    patch_evaluate,
    patch_launch,
    patch_train,
    patch_panda_env,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply targeted fixes to panda_rl")
    parser.add_argument("repo_root", type=pathlib.Path, help="Path to the panda_rl repository root")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        print(f"Repository path does not exist: {repo_root}", file=sys.stderr)
        return 1

    try:
        for patcher in PATCHERS:
            patcher(repo_root)
    except PatchError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Applied fixes successfully to: {repo_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
