"""Launch final fine-alignment SAC training for multiple random seeds."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2, 3]
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU ids assigned round-robin.",
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--train-args",
        default="",
        help="Additional arguments accepted by train_window_fine_align_sac.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timesteps <= 0 or args.n_envs <= 0:
        raise ValueError("--timesteps and --n-envs must be positive")
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")

    project_root = Path(__file__).resolve().parent
    train_script = project_root / "train" / "train_window_fine_align_sac.py"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_root = Path(
        args.output_root
        or project_root
        / "outputs"
        / "window_fine_align"
        / f"multi-seed-{timestamp}"
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    launched: list[tuple[int, str, int, Path]] = []
    for index, seed in enumerate(args.seeds):
        gpu = gpu_ids[index % len(gpu_ids)]
        run_dir = output_root / f"seed-{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            sys.executable,
            str(train_script),
            "--timesteps",
            str(args.timesteps),
            "--n-envs",
            str(args.n_envs),
            "--seed",
            str(seed),
            "--device",
            "cuda:0",
            "--output-dir",
            str(run_dir),
            *shlex.split(args.train_args),
        ]
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        launched.append((seed, gpu, process.pid, log_path))
        print(
            f"seed={seed} gpu={gpu} pid={process.pid} "
            f"device=cuda:0 log={log_path}"
        )

    print(f"Launched {len(launched)} runs under {output_root}")


if __name__ == "__main__":
    main()
