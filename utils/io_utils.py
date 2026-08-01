"""Explicit JSON and SAC model IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3 import SAC


def load_sac_model(
    model_path: str | Path,
    *,
    env: Any = None,
    device: str = "auto",
) -> SAC:
    """Load exactly the requested SAC artifact without configuration inference."""

    path = Path(model_path).expanduser()
    if not path.is_file() and path.suffix != ".zip":
        zip_path = path.with_suffix(".zip")
        if zip_path.is_file():
            path = zip_path
    if not path.is_file():
        raise FileNotFoundError(f"SAC model does not exist: {path}")
    return SAC.load(str(path), env=env, device=device)


def save_json(data: Any, path: str | Path) -> Path:
    """Write UTF-8 JSON, creating only the requested parent directory."""

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON from an explicit path."""

    input_path = Path(path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))
