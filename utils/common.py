"""Common configuration, logging, and reproducibility helpers."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the REIM project root."""
    value = Path(path).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update ``base`` and return it."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def seed_everything(seed: int, deterministic_torch: bool = False) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch, and CUDA RNG states for exact resume."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        state["cuda"] = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
    except ImportError:
        state["torch"] = None
        state["cuda"] = None
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore an RNG snapshot created by :func:`capture_rng_state`."""

    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(tuple(state["numpy"]))
    try:
        import torch

        if state.get("torch") is not None:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except ImportError:
        pass


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def configure_logging(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if log_file is not None:
        target = resolve_path(log_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def ensure_layout() -> None:
    for relative in (
        "datasets/demonstrations",
        "datasets/failures",
        "checkpoints",
        "results/tables",
        "results/figures",
        "results/logs",
        "paper_assets",
    ):
        resolve_path(relative).mkdir(parents=True, exist_ok=True)
