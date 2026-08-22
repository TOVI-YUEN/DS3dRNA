# DS3dRNA/potential.py

from typing import List, Optional

import torch

try:
    from .tensor_loader import load_tensor, resolve_tensor_path
except ImportError:
    from tensor_loader import load_tensor, resolve_tensor_path


class TriRNASP_Potential:
    """
    Container for TriRNASP energy tensors.

    Supports both the original text energy files (*.energy) and NumPy binary
    files (*.npy). A .npy file may be either:
      - np.loadtxt("*.energy") saved with np.save, i.e. a table with >= 7 columns
      - a 1D vector of values in original file order
      - a materialized 6D tensor
    """

    def __init__(
        self,
        rough_path: Optional[str] = None,
        fine_path: Optional[str] = None,
        stage_paths: Optional[List[str]] = None,
        bw_list: Optional[List[float]] = None,
        D: int = 12,
        R0: float = 8.0,
        bw_rough: float = 1.33333,
        bw_fine: float = 0.57143,
        device=None,
        prefer_npy: bool = False,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.D = D
        self.R0 = R0
        self.energy_paths: List[str] = []

        if stage_paths is not None:
            if bw_list is None:
                raise ValueError("bw_list is required when stage_paths is provided.")
            if len(stage_paths) == 0:
                raise ValueError("stage_paths cannot be empty.")
            if len(stage_paths) != len(bw_list):
                raise ValueError("stage_paths and bw_list must have the same length.")

            self.n_stage = len(stage_paths)
            self.energy_list: List[torch.Tensor] = []
            self.intervals_list: List[int] = []
            self.bw_list: List[float] = list(bw_list)

            for path, bw in zip(stage_paths, bw_list):
                T, intervals, resolved = self._load_one(path, D=D, R0=R0, bw=bw, device=device, prefer_npy=prefer_npy)
                self.energy_list.append(T)
                self.intervals_list.append(intervals)
                self.energy_paths.append(resolved)

            self.energy_rough = self.energy_list[0]
            self.intervals_rough = self.intervals_list[0]
            self.bw_rough = self.bw_list[0]

            self.energy_fine = self.energy_list[-1]
            self.intervals_fine = self.intervals_list[-1]
            self.bw_fine = self.bw_list[-1]
            self.rough_path = self.energy_paths[0]
            self.fine_path = self.energy_paths[-1]
            return

        if rough_path is not None and fine_path is not None:
            self.energy_rough, self.intervals_rough, self.rough_path = self._load_one(
                rough_path,
                D=D,
                R0=R0,
                bw=bw_rough,
                device=device,
                prefer_npy=prefer_npy,
            )
            self.energy_fine, self.intervals_fine, self.fine_path = self._load_one(
                fine_path,
                D=D,
                R0=R0,
                bw=bw_fine,
                device=device,
                prefer_npy=prefer_npy,
            )

            self.bw_rough = bw_rough
            self.bw_fine = bw_fine

            self.n_stage = 2
            self.energy_list = [self.energy_rough, self.energy_fine]
            self.intervals_list = [self.intervals_rough, self.intervals_fine]
            self.bw_list = [self.bw_rough, self.bw_fine]
            self.energy_paths = [self.rough_path, self.fine_path]
            return

        raise ValueError(
            "TriRNASP_Potential initialization is ambiguous: provide either "
            "rough_path and fine_path, or stage_paths and bw_list."
        )

    @staticmethod
    def _load_one(path: str, D: int, R0: float, bw: float, device, prefer_npy: bool):
        resolved = resolve_tensor_path(path, prefer_npy=prefer_npy)
        np_arr, intervals = load_tensor(resolved, D=D, R0_orig=R0, bin_width=bw)
        return torch.from_numpy(np_arr).to(device=device), intervals, resolved
