"""Dataset utilities and causal failure annotation for REIM."""

from .failure_labels import (
    CausalFailureLabeler,
    build_causal_windows,
    build_prospective_failure_targets,
)
from .io import (
    atomic_save_npz,
    atomic_write_json,
    file_sha256,
    json_compatible,
)

__all__ = [
    "CausalFailureLabeler",
    "build_causal_windows",
    "build_prospective_failure_targets",
    "atomic_save_npz",
    "atomic_write_json",
    "file_sha256",
    "json_compatible",
]
