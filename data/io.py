"""Safe, deterministic IO helpers for trajectory datasets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _replace_with_retry(source: str, destination: Path) -> None:
    """Atomic rename that tolerates transient Windows file locks.

    Windows Defender, search indexing, and sync clients can briefly hold a
    lock on a freshly written file, making ``os.replace`` fail with
    ``PermissionError`` even though the operation is safe to retry.
    """

    attempts = 8
    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


def json_compatible(value: Any) -> Any:
    """Recursively convert NumPy/path values into strict JSON values."""

    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically write stable, human-readable JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                json_compatible(payload),
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def atomic_save_npz(path: str | Path, **arrays: Any) -> Path:
    """Atomically save a compressed NPZ without pickled object arrays."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=".npz",
        dir=destination.parent,
    )
    os.close(fd)
    try:
        normalized: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            array = np.asarray(value)
            if array.dtype == object:
                raise TypeError(
                    f"Refusing to pickle object dtype for NPZ key {key!r}; "
                    "use a fixed-width Unicode/bytes array."
                )
            normalized[key] = array
        np.savez_compressed(temporary_name, **normalized)
        # Flush to durable storage with a writable handle: Windows cannot
        # fsync a read-only file descriptor, POSIX accepts both.
        with open(temporary_name, "r+b") as handle:
            os.fsync(handle.fileno())
        _replace_with_retry(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

