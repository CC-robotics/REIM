"""Robust NPZ readers shared by the supervised REIM trainers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


STATE_KEYS = ("states", "observations", "obs", "state")
ACTION_KEYS = ("actions", "action", "expert_actions")
LABEL_KEYS = ("labels", "failure_labels", "failure_label", "failures", "failure", "y")
WINDOW_KEYS = ("windows", "state_sequences", "sequences", "history_states")
LENGTH_KEYS = ("valid_lengths", "lengths", "sequence_lengths")


@dataclass(frozen=True)
class DemonstrationData:
    states: np.ndarray
    actions: np.ndarray
    groups: np.ndarray
    files: tuple[Path, ...]


@dataclass(frozen=True)
class FailureData:
    windows: np.ndarray
    labels: np.ndarray
    lengths: np.ndarray
    groups: np.ndarray
    files: tuple[Path, ...]


def discover_npz(path: str | Path) -> list[Path]:
    source = Path(path).expanduser()
    if source.is_file():
        if source.suffix.lower() != ".npz":
            raise ValueError(f"Expected an .npz file, got {source}")
        return [source]
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")
    files = sorted(source.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {source}")
    return files


def _first_key(archive: np.lib.npyio.NpzFile, keys: Iterable[str]) -> str | None:
    return next((key for key in keys if key in archive.files), None)


def _as_trajectories(array: np.ndarray) -> list[np.ndarray]:
    """Convert dense or object arrays to a list of [time, feature] arrays."""
    array = np.asarray(array)
    if array.dtype == object:
        return [
            np.asarray(item)
            if np.asarray(item).ndim > 1
            else np.asarray(item).reshape(1, -1)
            for item in array.reshape(-1)
        ]
    if array.ndim == 1:
        return [array.reshape(1, -1)]
    if array.ndim == 2:
        return [array]
    if array.ndim >= 3:
        return [array[index].reshape(-1, array.shape[-1]) for index in range(array.shape[0])]
    raise ValueError(f"Cannot interpret array with shape {array.shape} as trajectories.")


def load_demonstrations(path: str | Path) -> DemonstrationData:
    files = discover_npz(path)
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    group_id = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=True) as archive:
            state_key = _first_key(archive, STATE_KEYS)
            action_key = _first_key(archive, ACTION_KEYS)
            if state_key is None or action_key is None:
                continue
            state_trajectories = _as_trajectories(archive[state_key])
            action_trajectories = _as_trajectories(archive[action_key])
            if len(state_trajectories) != len(action_trajectories):
                # A flat file is still usable when both sides have the same rows.
                states = np.asarray(archive[state_key]).reshape(
                    -1, np.asarray(archive[state_key]).shape[-1]
                )
                actions = np.asarray(archive[action_key]).reshape(
                    -1, np.asarray(archive[action_key]).shape[-1]
                )
                state_trajectories, action_trajectories = [states], [actions]

            for states, actions in zip(state_trajectories, action_trajectories):
                states = np.asarray(states, dtype=np.float32)
                actions = np.asarray(actions, dtype=np.float32)
                count = min(len(states), len(actions))
                if count == 0:
                    group_id += 1
                    continue
                states, actions = states[:count], actions[:count]
                finite = np.isfinite(states).all(axis=1) & np.isfinite(actions).all(axis=1)
                states, actions = states[finite], actions[finite]
                if len(states):
                    all_states.append(states)
                    all_actions.append(actions)
                    all_groups.append(np.full(len(states), group_id, dtype=np.int64))
                group_id += 1

    if not all_states:
        raise ValueError(
            f"No valid state/action pairs found in {path}. "
            f"Expected keys {STATE_KEYS} and {ACTION_KEYS}."
        )
    state_dims = {array.shape[-1] for array in all_states}
    action_dims = {array.shape[-1] for array in all_actions}
    if len(state_dims) != 1 or len(action_dims) != 1:
        raise ValueError(
            f"Inconsistent dimensions in demonstrations: states={state_dims}, "
            f"actions={action_dims}."
        )
    return DemonstrationData(
        states=np.concatenate(all_states).astype(np.float32, copy=False),
        actions=np.concatenate(all_actions).astype(np.float32, copy=False),
        groups=np.concatenate(all_groups),
        files=tuple(files),
    )


def make_causal_windows(
    states: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad causal histories so valid frames work with packed LSTMs."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"Expected [time, state_dim] states, got {states.shape}.")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    windows = np.zeros(
        (len(states), sequence_length, states.shape[-1]), dtype=np.float32
    )
    lengths = np.empty(len(states), dtype=np.int64)
    for timestep in range(len(states)):
        start = max(0, timestep - sequence_length + 1)
        history = states[start : timestep + 1]
        lengths[timestep] = len(history)
        windows[timestep, : len(history)] = history
    return windows, lengths


def _labels_for_trajectory(labels: np.ndarray, index: int, count: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.dtype == object:
        selected = np.asarray(labels.reshape(-1)[index]).reshape(-1)
    elif labels.ndim == 0:
        selected = np.repeat(labels.reshape(1), count)
    elif labels.ndim == 1:
        selected = labels.reshape(-1)
    else:
        selected = np.asarray(labels[index]).reshape(-1)
    if selected.size == 1 and count != 1:
        selected = np.repeat(selected, count)
    return selected[:count]


def _canonicalize_existing_windows(
    windows: np.ndarray,
    lengths: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim == 2:
        windows = windows[None, ...]
    if windows.ndim != 3:
        raise ValueError(f"Expected windows [samples,time,state], got {windows.shape}.")
    result = np.zeros(
        (len(windows), sequence_length, windows.shape[-1]), dtype=np.float32
    )
    canonical_lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if canonical_lengths.size == 1 and len(windows) > 1:
        canonical_lengths = np.repeat(canonical_lengths, len(windows))
    canonical_lengths = np.clip(canonical_lengths[: len(windows)], 1, sequence_length)
    for index, (window, length) in enumerate(zip(windows, canonical_lengths)):
        length = min(int(length), len(window), sequence_length)
        # Existing datasets usually left-pad histories. Select the side with
        # larger energy, then move valid frames to the front for pack_padded_sequence.
        if length < len(window):
            leading = window[:length]
            trailing = window[-length:]
            valid = trailing if np.linalg.norm(trailing) > np.linalg.norm(leading) else leading
        else:
            valid = window[-length:]
        result[index, :length] = valid[-length:]
    return result, canonical_lengths


def load_failure_data(
    path: str | Path,
    sequence_length: int = 10,
) -> FailureData:
    files = discover_npz(path)
    all_windows: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    group_id = 0

    for file_path in files:
        with np.load(file_path, allow_pickle=True) as archive:
            label_key = _first_key(archive, LABEL_KEYS)
            state_key = _first_key(archive, STATE_KEYS)
            window_key = _first_key(archive, WINDOW_KEYS)
            if label_key is None or (state_key is None and window_key is None):
                continue
            raw_labels = archive[label_key]

            # States are preferred: rebuilding their causal histories guarantees
            # a consistent padding convention across generated and external data.
            if state_key is not None:
                trajectories = _as_trajectories(archive[state_key])
                for trajectory_index, states in enumerate(trajectories):
                    states = np.asarray(states, dtype=np.float32)
                    labels = _labels_for_trajectory(
                        raw_labels, trajectory_index, len(states)
                    )
                    count = min(len(states), len(labels))
                    if count == 0:
                        group_id += 1
                        continue
                    states = states[:count]
                    labels = labels[:count]
                    windows, lengths = make_causal_windows(states, sequence_length)
                    finite = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(labels)
                    windows = windows[finite]
                    lengths = lengths[finite]
                    labels = labels[finite]
                    if len(windows):
                        all_windows.append(windows)
                        all_lengths.append(lengths)
                        all_labels.append(labels.astype(np.float32))
                        all_groups.append(
                            np.full(len(windows), group_id, dtype=np.int64)
                        )
                    group_id += 1
                continue

            raw_windows = np.asarray(archive[window_key])
            if raw_windows.dtype == object:
                window_trajectories = [
                    np.asarray(item) for item in raw_windows.reshape(-1)
                ]
            elif raw_windows.ndim == 4:
                window_trajectories = [
                    raw_windows[index] for index in range(raw_windows.shape[0])
                ]
            else:
                window_trajectories = [raw_windows]
            length_key = _first_key(archive, LENGTH_KEYS)
            raw_lengths = (
                np.asarray(archive[length_key])
                if length_key is not None
                else np.asarray(sequence_length)
            )
            for trajectory_index, windows in enumerate(window_trajectories):
                windows = np.asarray(windows)
                count = len(windows) if windows.ndim >= 3 else 1
                labels = _labels_for_trajectory(raw_labels, trajectory_index, count)
                if raw_lengths.ndim <= 1:
                    lengths = raw_lengths.reshape(-1)
                else:
                    lengths = np.asarray(raw_lengths[trajectory_index]).reshape(-1)
                if lengths.size == 1:
                    lengths = np.repeat(lengths, count)
                windows, lengths = _canonicalize_existing_windows(
                    windows, lengths, sequence_length
                )
                count = min(len(windows), len(labels), len(lengths))
                windows, labels, lengths = (
                    windows[:count],
                    labels[:count],
                    lengths[:count],
                )
                finite = np.isfinite(windows).all(axis=(1, 2)) & np.isfinite(labels)
                windows, labels, lengths = (
                    windows[finite],
                    labels[finite],
                    lengths[finite],
                )
                if len(windows):
                    all_windows.append(windows)
                    all_labels.append(labels.astype(np.float32))
                    all_lengths.append(lengths.astype(np.int64))
                    all_groups.append(np.full(len(windows), group_id, dtype=np.int64))
                group_id += 1

    if not all_windows:
        raise ValueError(
            f"No valid failure examples found in {path}. Expected labels {LABEL_KEYS} "
            f"and either states {STATE_KEYS} or windows {WINDOW_KEYS}."
        )
    state_dims = {array.shape[-1] for array in all_windows}
    if len(state_dims) != 1:
        raise ValueError(f"Inconsistent state dimensions in failure data: {state_dims}.")
    labels = np.concatenate(all_labels).astype(np.float32, copy=False)
    # Labels may arrive as bool, integer, probability, or {-1,+1}.
    labels = (labels > 0.5).astype(np.float32)
    return FailureData(
        windows=np.concatenate(all_windows).astype(np.float32, copy=False),
        labels=labels,
        lengths=np.concatenate(all_lengths).astype(np.int64, copy=False),
        groups=np.concatenate(all_groups),
        files=tuple(files),
    )


def group_train_validation_split(
    groups: np.ndarray,
    validation_fraction: float,
    seed: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by trajectory to avoid temporal leakage between train and validation."""
    groups = np.asarray(groups)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1.")
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    if len(unique_groups) >= 2:
        shuffled = rng.permutation(unique_groups)
        validation_count = int(round(len(shuffled) * validation_fraction))
        validation_count = min(max(validation_count, 1), len(shuffled) - 1)
        validation_groups = shuffled[:validation_count]
        validation_mask = np.isin(groups, validation_groups)
        train_indices = np.flatnonzero(~validation_mask)
        validation_indices = np.flatnonzero(validation_mask)
    else:
        indices = rng.permutation(len(groups))
        validation_count = min(
            max(int(round(len(indices) * validation_fraction)), 1),
            max(len(indices) - 1, 1),
        )
        validation_indices = indices[:validation_count]
        train_indices = indices[validation_count:]

    if labels is not None and len(np.unique(labels)) > 1:
        # A group-wise split is preferred; only fall back to a stratified row
        # split when it accidentally removes an entire class from either side.
        train_classes = np.unique(labels[train_indices])
        validation_classes = np.unique(labels[validation_indices])
        if len(train_classes) < 2 or len(validation_classes) < 2:
            train_parts: list[np.ndarray] = []
            validation_parts: list[np.ndarray] = []
            for class_value in np.unique(labels):
                class_indices = rng.permutation(np.flatnonzero(labels == class_value))
                validation_count = min(
                    max(int(round(len(class_indices) * validation_fraction)), 1),
                    max(len(class_indices) - 1, 1),
                )
                validation_parts.append(class_indices[:validation_count])
                train_parts.append(class_indices[validation_count:])
            train_indices = rng.permutation(np.concatenate(train_parts))
            validation_indices = rng.permutation(np.concatenate(validation_parts))

    if len(train_indices) == 0 or len(validation_indices) == 0:
        raise ValueError("Dataset is too small to create non-empty train/validation sets.")
    return train_indices, validation_indices
