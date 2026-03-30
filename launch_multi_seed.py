"""
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

    print("
All jobs launched.")
    for seed, gpu, pid, log_path in procs:
        print(f"  seed={seed} gpu={gpu} pid={pid} log={log_path}")


if __name__ == "__main__":
    main()
