#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Optional, Tuple

import torch
from .structure_fixed import RNA_Structure_Fixed
from .potential import TriRNASP_Potential

# dev: Apr-23 2026


# =========================
#  Helper functions for KL
# =========================

def estimate_prior_from_seqs(
    seqs: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Estimate a per-position categorical prior q(b | i) from a batch of sequences.

    Args:
        seqs:  (B, N_res) long tensor with values in {0,1,2,3}
        alpha: Laplace smoothing parameter

    Returns:
        probs: (4, N_res) float tensor with q(b | i)
    """
    if seqs.dim() == 1:
        seqs = seqs.unsqueeze(0)

    onehot = torch.nn.functional.one_hot(seqs.long(), num_classes=4)   # (B, N, 4)
    counts = onehot.sum(dim=0).permute(1, 0).to(torch.float32)         # (4, N)

    probs = (counts + alpha) / (counts.sum(dim=0, keepdim=True) + 4.0 * alpha)
    return probs  # (4, N)


def kl_penalty_from_prior(
    seqs: torch.Tensor,
    prior_probs: torch.Tensor,
    kl_lambda: float,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Sequence-prior KL regularization used in the new design code:

        KL_penalty(s) = λ * [ - (1/N) * Σ_i log q( s_i | i ) ]
    """
    if seqs.dim() == 1:
        seqs = seqs.unsqueeze(0)

    device = seqs.device
    B, N = seqs.shape
    eps = 1e-8

    log_q = torch.log(prior_probs + eps)  # (4, N)

    pos = torch.arange(N, device=device).unsqueeze(0).expand(B, N)  # (B, N)
    log_q_s = log_q[seqs, pos]  # (B, N)

    nll = -log_q_s.sum(dim=1)  # (B,)
    if normalize:
        nll = nll / float(N)

    return kl_lambda * nll


# =========================
#      TriRNASP Scorer
# =========================

class TriRNASP_Scorer:
    """
    Hybrid scorer:

    - If Tn (triplets for a stage) <= chunk_size:
        use the old fast path (expand)
        -> maximal throughput for normal sizes

    - If Tn > chunk_size:
        use the new chunked path
        -> avoids OOM for huge multi-state triplet counts

    Unified legality logic:
    1) Rough stage:
       - ALWAYS mask invalid atom/base combinations
       - do NOT add ss penalty

    2) Fine stage:
       - ALWAYS mask invalid atom/base combinations
       - if ss_constraint is None:
           * no extra penalties
       - if ss_constraint is not None:
           * add ss penalty only

    Important:
    - scorer reconstructs atom codes with the SAME legality semantics as
      structure_fixed.compute_code():
        C4' -> 0..3
        N1/N9 -> 4..7 with legality subsets
        P -> 8..11
        illegal N1/N9-base combos -> invalid
    - invalid codes NEVER contribute table energy in any stage
    """

    def __init__(
        self,
        potential: TriRNASP_Potential,
        structure: RNA_Structure_Fixed,
        device=None,
        R0_orig: float = 8.0,
        bw_rough: float = 1.33333,
        bw_fine: float = 0.57143,
        kl_coeff: float = 1.0,
        chunk_size: int = 300000,
        ss_constraint: Optional[Any] = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.pot = potential
        self.struct = structure
        self.kl_coeff = float(kl_coeff)
        self.chunk_size = int(chunk_size)
        self.ss_constraint = ss_constraint

        self.N_res = structure.N_res
        self.N_atoms = structure.N_atoms

        # Atom-level information
        self.atom_res = structure.atom_res_indices.to(device=device)     # (N_atoms,)
        self.atom_family = structure.atom_family.to(device=device)       # 0=C4',1=N1,2=N9,3=P
        self.coords = structure.coords.to(device=device)                 # (N_atoms,3)
        self.triplets = structure.triplets.to(device=device)             # (T,3)

        # Cache atom-family subsets once
        self.idx_C4 = (self.atom_family == 0).nonzero(as_tuple=False).squeeze(1)
        self.idx_N1 = (self.atom_family == 1).nonzero(as_tuple=False).squeeze(1)
        self.idx_N9 = (self.atom_family == 2).nonzero(as_tuple=False).squeeze(1)
        self.idx_P  = (self.atom_family == 3).nonzero(as_tuple=False).squeeze(1)

        # Precompute distances for all triplets once
        self._precompute_distances()

        # Prepare bins & trip lists
        if getattr(self.pot, "n_stage", 2) > 2:
            self._prepare_bins_multistage()
        else:
            self._prepare_bins_stage(
                R0_orig=R0_orig,
                bw_rough=bw_rough,
                bw_fine=bw_fine,
            )

    # ----------------------------------------------------------------------
    #  Utility
    # ----------------------------------------------------------------------

    def _normalize_seq_batch(self, seq_batch: torch.Tensor) -> torch.Tensor:
        """
        Accept (N,) or (B,N), return (B,N) long tensor on self.device.
        """
        if not isinstance(seq_batch, torch.Tensor):
            seq_batch = torch.as_tensor(seq_batch, dtype=torch.long, device=self.device)
        else:
            seq_batch = seq_batch.to(device=self.device)

        if seq_batch.dim() == 1:
            seq_batch = seq_batch.unsqueeze(0)
        elif seq_batch.dim() != 2:
            raise ValueError(f"seq_batch must be 1D or 2D, got shape={tuple(seq_batch.shape)}")

        return seq_batch.long()

    def _apply_ss_penalty(self, seq_batch: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        """
        Add optional sequence-level secondary-structure penalty.
        Fine-stage only.
        """
        if self.ss_constraint is None:
            return E

        E_ss = self.ss_constraint.score_batch(seq_batch)
        if not isinstance(E_ss, torch.Tensor):
            E_ss = torch.as_tensor(E_ss, dtype=E.dtype, device=E.device)
        else:
            E_ss = E_ss.to(device=E.device, dtype=E.dtype)

        if E_ss.dim() == 0:
            E_ss = E_ss.unsqueeze(0)

        if E_ss.shape != E.shape:
            raise ValueError(
                f"ss_constraint.score_batch returned shape {tuple(E_ss.shape)}, "
                f"but expected {tuple(E.shape)}"
            )

        return E + E_ss

    # ----------------------------------------------------------------------
    #  Precompute distances between atoms in each triplet
    # ----------------------------------------------------------------------

    def _precompute_distances(self):
        if self.triplets.numel() == 0:
            self.d12 = torch.empty((0,), device=self.device)
            self.d13 = torch.empty((0,), device=self.device)
            self.d23 = torch.empty((0,), device=self.device)
            return

        coords = self.coords
        trip = self.triplets

        i = trip[:, 0]
        j = trip[:, 1]
        k = trip[:, 2]

        v_i = coords[i]
        v_j = coords[j]
        v_k = coords[k]

        self.d12 = torch.linalg.norm(v_i - v_j, dim=1)
        self.d13 = torch.linalg.norm(v_i - v_k, dim=1)
        self.d23 = torch.linalg.norm(v_j - v_k, dim=1)

    # ----------------------------------------------------------------------
    #  2-stage mode: prepare Rough / Fine bins (legacy interface)
    # ----------------------------------------------------------------------

    def _prepare_bins_stage(self, R0_orig: float, bw_rough: float, bw_fine: float):
        device = self.device

        if self.d12.numel() == 0:
            self.trip_r = torch.empty((0, 3), dtype=torch.long, device=device)
            self.b12_r = self.b13_r = self.b23_r = torch.empty((0,), dtype=torch.long, device=device)
            self.trip_f = torch.empty((0, 3), dtype=torch.long, device=device)
            self.b12_f = self.b13_f = self.b23_f = torch.empty((0,), dtype=torch.long, device=device)

            self.trip_list = [self.trip_r, self.trip_f]
            self.b12_list = [self.b12_r, self.b12_f]
            self.b13_list = [self.b13_r, self.b13_f]
            self.b23_list = [self.b23_r, self.b23_f]
            return

        # Rough
        I_r = self.pot.intervals_rough
        b12_r_all = torch.floor(self.d12 / bw_rough).long()
        b13_r_all = torch.floor(self.d13 / bw_rough).long()
        b23_r_all = torch.floor(self.d23 / bw_rough).long()

        mask_r = (
            (b12_r_all >= 0) & (b12_r_all < I_r) &
            (b13_r_all >= 0) & (b13_r_all < I_r) &
            (b23_r_all >= 0) & (b23_r_all < 2 * I_r)
        )

        self.trip_r = self.triplets[mask_r]
        self.b12_r = b12_r_all[mask_r]
        self.b13_r = b13_r_all[mask_r]
        self.b23_r = b23_r_all[mask_r]

        # Fine
        I_f = self.pot.intervals_fine
        b12_f_all = torch.floor(self.d12 / bw_fine).long()
        b13_f_all = torch.floor(self.d13 / bw_fine).long()
        b23_f_all = torch.floor(self.d23 / bw_fine).long()

        mask_f = (
            (b12_f_all >= 0) & (b12_f_all < I_f) &
            (b13_f_all >= 0) & (b13_f_all < I_f) &
            (b23_f_all >= 0) & (b23_f_all < 2 * I_f)
        )

        self.trip_f = self.triplets[mask_f]
        self.b12_f = b12_f_all[mask_f]
        self.b13_f = b13_f_all[mask_f]
        self.b23_f = b23_f_all[mask_f]

        self.trip_list = [self.trip_r, self.trip_f]
        self.b12_list = [self.b12_r, self.b12_f]
        self.b13_list = [self.b13_r, self.b13_f]
        self.b23_list = [self.b23_r, self.b23_f]

    # ----------------------------------------------------------------------
    #  Multi-stage mode: prepare bins for Stage_1 ... Stage_N
    # ----------------------------------------------------------------------

    def _prepare_bins_multistage(self):
        device = self.device

        if self.d12.numel() == 0:
            self.trip_list, self.b12_list, self.b13_list, self.b23_list = [], [], [], []
            self.trip_r = torch.empty((0, 3), dtype=torch.long, device=device)
            self.b12_r = self.b13_r = self.b23_r = torch.empty((0,), dtype=torch.long, device=device)
            self.trip_f = self.trip_r
            self.b12_f = self.b12_r
            self.b13_f = self.b13_r
            self.b23_f = self.b23_r
            return

        self.trip_list, self.b12_list, self.b13_list, self.b23_list = [], [], [], []

        for bw, intervals in zip(self.pot.bw_list, self.pot.intervals_list):
            b12_all = torch.floor(self.d12 / bw).long()
            b13_all = torch.floor(self.d13 / bw).long()
            b23_all = torch.floor(self.d23 / bw).long()

            mask = (
                (b12_all >= 0) & (b12_all < intervals) &
                (b13_all >= 0) & (b13_all < intervals) &
                (b23_all >= 0) & (b23_all < 2 * intervals)
            )

            self.trip_list.append(self.triplets[mask].to(device))
            self.b12_list.append(b12_all[mask].to(device))
            self.b13_list.append(b13_all[mask].to(device))
            self.b23_list.append(b23_all[mask].to(device))

        self.trip_r = self.trip_list[0]
        self.b12_r = self.b12_list[0]
        self.b13_r = self.b13_list[0]
        self.b23_r = self.b23_list[0]

        self.trip_f = self.trip_list[-1]
        self.b12_f = self.b12_list[-1]
        self.b13_f = self.b13_list[-1]
        self.b23_f = self.b23_list[-1]

    # ----------------------------------------------------------------------
    #  Rebuild codes with EXACT structure_fixed legality semantics
    # ----------------------------------------------------------------------

    def _codes_and_valid_from_seq(self, seq_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        seq_batch = self._normalize_seq_batch(seq_batch)

        B, N_res = seq_batch.shape
        if N_res != self.N_res:
            raise AssertionError(f"sequence length must equal N_res ({self.N_res}), got {N_res}")

        base_idx = seq_batch[:, self.atom_res]  # (B, N_atoms)
        codes = torch.full((B, self.N_atoms), -1, dtype=torch.long, device=self.device)
        valid_atom = torch.zeros((B, self.N_atoms), dtype=torch.bool, device=self.device)

        # C4' -> 0..3
        if self.idx_C4.numel() > 0:
            bsub = base_idx[:, self.idx_C4]
            codes[:, self.idx_C4] = bsub
            valid_atom[:, self.idx_C4] = True

        # P -> 8..11
        if self.idx_P.numel() > 0:
            bsub = base_idx[:, self.idx_P]
            codes[:, self.idx_P] = 8 + bsub
            valid_atom[:, self.idx_P] = True

        # N1 -> 4..7, only U/C legal
        if self.idx_N1.numel() > 0:
            bsub = base_idx[:, self.idx_N1]
            valid_n1 = (bsub == 1) | (bsub == 2)
            codes[:, self.idx_N1] = torch.where(
                valid_n1,
                4 + bsub,
                torch.full_like(bsub, -1),
            )
            valid_atom[:, self.idx_N1] = valid_n1

        # N9 -> 4..7, only A/G legal
        if self.idx_N9.numel() > 0:
            bsub = base_idx[:, self.idx_N9]
            valid_n9 = (bsub == 0) | (bsub == 3)
            codes[:, self.idx_N9] = torch.where(
                valid_n9,
                4 + bsub,
                torch.full_like(bsub, -1),
            )
            valid_atom[:, self.idx_N9] = valid_n9

        # Safe indexing: invalid -> 0, but masked out later
        codes_safe = torch.where(valid_atom, codes, torch.zeros_like(codes))
        return codes_safe, valid_atom

    def _codes_from_seq(self, seq_batch: torch.Tensor) -> torch.Tensor:
        codes_safe, _ = self._codes_and_valid_from_seq(seq_batch)
        return codes_safe

    def _valid_atom_mask(self, seq_batch: torch.Tensor) -> torch.Tensor:
        _, valid_atom = self._codes_and_valid_from_seq(seq_batch)
        return valid_atom

    # ----------------------------------------------------------------------
    #  OLD fast path
    # ----------------------------------------------------------------------

    def _energy_stage_old_fast(self, seq_batch: torch.Tensor, stage_idx: int) -> torch.Tensor:
        seq_batch = self._normalize_seq_batch(seq_batch)
        codes, valid_atom = self._codes_and_valid_from_seq(seq_batch)
        B = codes.shape[0]

        trip = self.trip_list[stage_idx]
        b12 = self.b12_list[stage_idx]
        b13 = self.b13_list[stage_idx]
        b23 = self.b23_list[stage_idx]
        T = self.pot.energy_list[stage_idx]

        if trip.numel() == 0:
            return torch.zeros((B,), dtype=torch.float32, device=self.device)

        i = trip[:, 0]
        j = trip[:, 1]
        k = trip[:, 2]
        Tn = trip.shape[0]

        ci = codes[:, i]
        cj = codes[:, j]
        ck = codes[:, k]

        b12_b = b12.unsqueeze(0).expand(B, Tn)
        b13_b = b13.unsqueeze(0).expand(B, Tn)
        b23_b = b23.unsqueeze(0).expand(B, Tn)

        vals = T[ci, cj, ck, b12_b, b13_b, b23_b]

        # ALWAYS mask invalid triplets, regardless of --ss
        valid_trip = valid_atom[:, i] & valid_atom[:, j] & valid_atom[:, k]
        vals = vals * valid_trip.to(vals.dtype)

        return vals.sum(dim=1)

    # ----------------------------------------------------------------------
    #  NEW chunked path
    # ----------------------------------------------------------------------

    def _energy_stage_chunked(self, seq_batch: torch.Tensor, stage_idx: int) -> torch.Tensor:
        seq_batch = self._normalize_seq_batch(seq_batch)
        codes, valid_atom = self._codes_and_valid_from_seq(seq_batch)
        B = codes.shape[0]

        trip = self.trip_list[stage_idx]
        b12 = self.b12_list[stage_idx]
        b13 = self.b13_list[stage_idx]
        b23 = self.b23_list[stage_idx]
        T = self.pot.energy_list[stage_idx]

        if trip.numel() == 0:
            return torch.zeros((B,), dtype=torch.float32, device=self.device)

        i_all = trip[:, 0]
        j_all = trip[:, 1]
        k_all = trip[:, 2]
        Tn = trip.shape[0]

        E = torch.zeros((B,), dtype=torch.float32, device=self.device)

        cs = self.chunk_size
        for t0 in range(0, Tn, cs):
            t1 = min(Tn, t0 + cs)

            i = i_all[t0:t1]
            j = j_all[t0:t1]
            k = k_all[t0:t1]

            ci = codes[:, i]
            cj = codes[:, j]
            ck = codes[:, k]

            b12_c = b12[t0:t1]
            b13_c = b13[t0:t1]
            b23_c = b23[t0:t1]

            vals = T[
                ci, cj, ck,
                b12_c.unsqueeze(0),
                b13_c.unsqueeze(0),
                b23_c.unsqueeze(0),
            ]

            # ALWAYS mask invalid triplets, regardless of --ss
            valid_trip = valid_atom[:, i] & valid_atom[:, j] & valid_atom[:, k]
            vals = vals * valid_trip.to(vals.dtype)

            E += vals.sum(dim=1)

            del i, j, k, ci, cj, ck, vals, b12_c, b13_c, b23_c

        return E

    # ----------------------------------------------------------------------
    #  Public per-stage energy (hybrid switch)
    # ----------------------------------------------------------------------

    def energy_stage(self, seq_batch: torch.Tensor, stage_idx: int) -> torch.Tensor:
        seq_batch = self._normalize_seq_batch(seq_batch)

        trip = self.trip_list[stage_idx]
        Tn = int(trip.shape[0]) if trip.numel() else 0
        if Tn > self.chunk_size:
            return self._energy_stage_chunked(seq_batch, stage_idx)
        return self._energy_stage_old_fast(seq_batch, stage_idx)

    # ----------------------------------------------------------------------
    #  Internal single-stage scoring ("rough" / "fine")
    # ----------------------------------------------------------------------

    def _score_stage(
        self,
        seq_batch: torch.Tensor,
        stage: str,
        with_kl: bool = False,
        kl_lambda: float | None = None,
    ) -> torch.Tensor:
        seq_batch = self._normalize_seq_batch(seq_batch)

        if stage == "rough":
            stage_idx = 0
        elif stage == "fine":
            stage_idx = len(self.trip_list) - 1
        else:
            raise ValueError(f"Unknown stage: {stage}")

        E = self.energy_stage(seq_batch, stage_idx=stage_idx)

        if with_kl:
            if kl_lambda is None:
                kl_lambda = self.kl_coeff
            prior = estimate_prior_from_seqs(seq_batch)
            kl_penalty = kl_penalty_from_prior(
                seq_batch, prior_probs=prior, kl_lambda=kl_lambda, normalize=True
            )
            E = E + kl_penalty

        # Fine stage only: keep mask behavior unchanged, add SS penalty if enabled
        if stage == "fine" and self.ss_constraint is not None:
            E = self._apply_ss_penalty(seq_batch, E)

        return E

    # Public single-stage API
    def score_batch(
        self,
        seq_batch: torch.Tensor,
        stage: str = "fine",
        with_kl: bool = False,
        kl_lambda: float | None = None,
    ) -> torch.Tensor:
        seq_batch = self._normalize_seq_batch(seq_batch)
        return self._score_stage(seq_batch, stage=stage, with_kl=with_kl, kl_lambda=kl_lambda)

    # ----------------------------------------------------------------------
    #  Two-stage scoring: Rough -> Fine TOP-K (with sequence-prior KL)
    # ----------------------------------------------------------------------

    def score_batch_two_stage(
        self,
        seq_batch: torch.Tensor,
        top_k: int = 10,
        with_kl_rough: bool = True,
        kl_lambda: float | None = None,
    ):
        seq_batch = self._normalize_seq_batch(seq_batch)
        B = seq_batch.shape[0]
        k = min(top_k, B)

        if kl_lambda is None:
            kl_lambda = self.kl_coeff

        # Rough
        E_rough = self._score_stage(
            seq_batch,
            stage="rough",
            with_kl=False,
            kl_lambda=kl_lambda,
        )

        # Optional KL at rough
        if with_kl_rough and kl_lambda > 0.0:
            prior = estimate_prior_from_seqs(seq_batch)
            kl_penalty = kl_penalty_from_prior(
                seq_batch, prior_probs=prior, kl_lambda=kl_lambda, normalize=True
            )
            E_rough = E_rough + kl_penalty

        # TOP-K
        rough_top_vals, rough_top_idx = torch.topk(E_rough, k=k, largest=False)

        # Fine on TOP-K
        seq_top = seq_batch[rough_top_idx]
        E_fine_top = self._score_stage(
            seq_top,
            stage="fine",
            with_kl=False,
            kl_lambda=kl_lambda,
        )

        # Sort by fine
        fine_sorted_vals, order = torch.sort(E_fine_top, descending=False)
        rough_sorted_vals = rough_top_vals[order]
        idx_sorted = rough_top_idx[order]

        return fine_sorted_vals, rough_sorted_vals, idx_sorted
