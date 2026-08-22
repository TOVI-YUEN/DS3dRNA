# DS3dRNA/tensor_loader.py

import os
from typing import Iterable, Tuple

import numpy as np


def resolve_tensor_path(path, prefer_npy: bool = False) -> str:
    """
    Resolve an energy tensor path.

    Supports both the original text format (*.energy) and NumPy binary files
    (*.npy). If the requested suffix is missing, the sibling suffix is tried.
    """
    path = os.fspath(path)
    root, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext in {".energy", ".npy"}:
        npy_path = root + ".npy"
        energy_path = root + ".energy"
        candidates = [npy_path, energy_path] if prefer_npy else [path]
        candidates.extend([npy_path, energy_path])
    else:
        candidates = [path + ".npy", path + ".energy"] if prefer_npy else [path, path + ".energy", path + ".npy"]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            return candidate

    tried = ", ".join(dict.fromkeys(candidates))
    raise FileNotFoundError(f"Energy tensor file not found. Tried: {tried}")


def _compute_intervals(R0_orig: float, bin_width: float) -> int:
    intervals = int(np.floor(float(R0_orig) / float(bin_width) + 0.5))
    return max(1, intervals)


def _iter_valid_slots(D: int, intervals: int) -> Iterable[Tuple[int, int, int, int, int, int]]:
    for n1 in range(D):
        for n2 in range(D):
            for n3 in range(D):
                for n4 in range(intervals):
                    for n5 in range(intervals):
                        if abs(n4 - n5) == 0:
                            n6_min = 0
                        else:
                            n6_min = abs(n4 - n5) - 1
                        n6_max = n4 + n5 + 1

                        for n6 in range(n6_min, n6_max + 1):
                            yield n1, n2, n3, n4, n5, n6


def _expected_value_count(D: int, intervals: int) -> int:
    return sum(1 for _ in _iter_valid_slots(D, intervals))


def _empty_tensor(D: int, intervals: int) -> np.ndarray:
    return np.zeros((D, D, D, intervals, intervals, 2 * intervals), dtype=np.float32)


def _fill_tensor_from_ordered_values(values, path: str, D: int, intervals: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    expected = _expected_value_count(D, intervals)
    if values.size != expected:
        raise RuntimeError(
            f"Bad energy value count in {path}: got {values.size}, expected {expected} "
            f"(D={D}, intervals={intervals})"
        )

    T = _empty_tensor(D, intervals)
    for i, slot in enumerate(_iter_valid_slots(D, intervals)):
        T[slot] = values[i]
    return T


def _load_energy_text(path: str, D: int, intervals: int) -> np.ndarray:
    T = _empty_tensor(D, intervals)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for slot in _iter_valid_slots(D, intervals):
            line = f.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF in energy file {path}")
            parts = line.split()
            if len(parts) < 7:
                raise RuntimeError(f"Bad line in {path}: {line.rstrip()}")
            T[slot] = float(parts[6])

    return T


def _load_energy_npy(path: str, D: int, intervals: int) -> np.ndarray:
    arr = np.load(path, allow_pickle=False, mmap_mode="r")

    if arr.ndim == 6:
        expected_shape = (D, D, D, intervals, intervals, 2 * intervals)
        if tuple(arr.shape) != expected_shape:
            raise RuntimeError(
                f"Bad 6D tensor shape in {path}: got {tuple(arr.shape)}, expected {expected_shape}"
            )
        return np.array(arr, dtype=np.float32, copy=True)

    if arr.ndim == 2 and arr.shape[1] >= 7:
        return _fill_tensor_from_ordered_values(arr[:, 6], path, D, intervals)

    if arr.ndim == 1:
        return _fill_tensor_from_ordered_values(arr, path, D, intervals)

    raise RuntimeError(
        f"Unsupported .npy energy format in {path}: shape={tuple(arr.shape)}. "
        "Expected a 6D tensor, a 1D value vector, or the loadtxt table with at least 7 columns."
    )


def load_tensor(path, D=12, R0_orig=8.0, bin_width=1.0, prefer_npy: bool = False):
    """
    Load a TriRNASP 6D energy tensor.

    Accepted inputs:
      - *.energy: original text format, value in column 7
      - *.npy: np.loadtxt("*.energy") table saved with np.save
      - *.npy: 1D value vector in original file order
      - *.npy: already materialized 6D tensor

    Returns:
      T: np.ndarray, shape (D, D, D, intervals, intervals, 2 * intervals)
      intervals: int
    """
    intervals = _compute_intervals(R0_orig, bin_width)
    resolved_path = resolve_tensor_path(path, prefer_npy=prefer_npy)
    ext = os.path.splitext(resolved_path)[1].lower()

    if ext == ".npy":
        return _load_energy_npy(resolved_path, D, intervals), intervals
    if ext == ".energy":
        return _load_energy_text(resolved_path, D, intervals), intervals

    raise RuntimeError(f"Unsupported energy tensor suffix: {resolved_path}")
