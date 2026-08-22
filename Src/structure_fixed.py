#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DS3dRNA fixed-structure representation.

Robust + fast RNA pseudo-atom structure loader for TriRNASP / TriRNAde.

Design goals:
- Never crash on malformed PDB lines (short TER, truncated ATOM, wrong columns, etc.)
- Filter EVERYTHING that is not RNA-like residue (A/U/C/G/N; optional DNA DA/DG/DC/DT -> A/G/C/U)
- Keep only pseudo-atoms: C4', N1, N9, P (supports C4* -> C4')
- Count residues including unknown base 'N' as placeholder (native_base_idx = -1)
- Unknown base 'N' contributes NO atoms and NO energy

This file intentionally avoids Bio.PDB.PDBParser to prevent IndexError on broken PDB lines.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch

# ----------------- base encoding ----------------- #

BASE_TO_IDX = {"A": 0, "U": 1, "C": 2, "G": 3}
IDX_TO_BASE = {v: k for k, v in BASE_TO_IDX.items()}

# Optional: map common DNA residue names to RNA equivalents
DNA_TO_RNA = {"DA": "A", "DG": "G", "DC": "C", "DT": "U"}

# Treat these as "RNA residues" for counting N_res / res_keys.
# - A/U/C/G : valid bases
# - N       : unknown base placeholder (counted in length but invalid for embedding)
# - DA/DG/DC/DT (optional) -> mapped to A/G/C/U
RNA_LIKE = set(BASE_TO_IDX.keys()) | {"N"} | set(DNA_TO_RNA.keys())


# ----------- C4'/N1/N9/P compute_code (strict, aligned with C TriRNASP) ----------- #

def compute_code(resname: str, atom: str) -> int:
    """
    resname: "A","U","C","G"
    atom   : "C4'", "N1", "N9", "P"
    return: 0..11 or -1 (invalid)

    Mapping aligned with C version TriRNASP:
      base_idx: A=0,U=1,C=2,G=3

      C4'   -> 0..3
      N1/N9 -> 4..7  (N9 only A/G, N1 only U/C)
      P     -> 8..11
    """
    resname = resname.strip().upper()
    if resname not in BASE_TO_IDX:
        return -1
    base_idx = BASE_TO_IDX[resname]

    atom = atom.strip().upper()

    if atom == "C4'":
        return base_idx

    if atom == "P":
        return 8 + base_idx

    if atom == "N9":
        return (4 + base_idx) if resname in ("A", "G") else -1

    if atom == "N1":
        return (4 + base_idx) if resname in ("U", "C") else -1

    return -1


# ----------------- robust PDB parsing helpers ----------------- #

_KEEP_ATOMS = {"C4'", "N1", "N9", "P"}


def _norm_atom_name(aname: str) -> str:
    """Normalize atom name for compatibility (C4* -> C4')."""
    a = (aname or "").strip()
    if a == "C4*":
        return "C4'"
    return a


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _safe_float(x: str) -> Optional[float]:
    try:
        v = float(str(x).strip())
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _parse_pdb_atom_line(line: str) -> Optional[Tuple[str, str, str, int, str, float, float, float, str]]:
    """
    Robust ATOM/HETATM parser.

    Returns:
      (atom_name, resname, chain_id, resseq, icode, x, y, z, altloc)

    Or None if invalid/unparseable.
    """
    if not line.startswith(("ATOM", "HETATM")):
        return None

    raw = line.rstrip("\n")

    # ---- fixed-column parse (preferred) ----
    if len(raw) >= 54:
        atom_name = _norm_atom_name(raw[12:16])
        altloc = raw[16:17]  # may be ' ' or 'A'...
        resname = raw[17:20].strip().upper()
        chain_id = raw[21:22].strip() or " "
        resseq = _safe_int(raw[22:26])
        icode = raw[26:27] if len(raw) >= 27 else " "

        x = _safe_float(raw[30:38])
        y = _safe_float(raw[38:46])
        z = _safe_float(raw[46:54])

        if resseq is not None and x is not None and y is not None and z is not None:
            return atom_name, resname, chain_id, resseq, (icode if icode else " "), x, y, z, (altloc if altloc else " ")

    # ---- fallback: whitespace split parse ----
    parts = raw.split()
    # minimal: ATOM serial atom resname chain resseq x y z
    if len(parts) < 9:
        return None

    atom_name = _norm_atom_name(parts[2])
    resname = parts[3].upper()
    chain_id = parts[4] if parts[4] else " "
    resseq = _safe_int(parts[5])
    x = _safe_float(parts[6])
    y = _safe_float(parts[7])
    z = _safe_float(parts[8])

    if resseq is None or x is None or y is None or z is None:
        return None

    # split mode usually loses altloc/icode; assume blank
    return atom_name, resname, chain_id, resseq, " ", x, y, z, " "


# ----------------- main class ----------------- #

class RNA_Structure_Fixed:
    """
    Fixed structure representation for TriRNASP scorer:
      - keep only pseudo-atoms: C4', N1, N9, P (for valid bases A/U/C/G)
      - count residues (N_res) INCLUDING unknown base 'N' as placeholder
        (native_base_idx = -1), but N contributes NO atoms and NO energy.

    Robustness:
      - ignores all non-ATOM/HETATM lines (TER/END/REMARK won't crash)
      - drops malformed/truncated ATOM lines
      - filters proteins/ligands/ions by residue name
    """

    def __init__(self, pdb_path: str, R0: float = 8.0, bin_width: float = 0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        atoms = []
        atom_codes = []
        atom_res_indices = []
        atom_family = []  # 0=C4',1=N1,2=N9,3=P

        # Residue indexing: key=(chain_id, resseq, icode) -> 0..N_res-1
        res_keys = []
        res_key_to_idx = {}
        res_base_idx = []  # native base idx per residue: 0..3 valid, -1 unknown (N)

        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue

                parsed = _parse_pdb_atom_line(line)
                if parsed is None:
                    continue

                aname, resname, chain_id, resseq, icode, x, y, z, altloc = parsed

                # Filter altLoc: keep blank or 'A' only
                if altloc not in (" ", "", "A"):
                    continue

                # DNA -> RNA mapping (optional)
                if resname in DNA_TO_RNA:
                    resname = DNA_TO_RNA[resname]

                # Keep only RNA-like residues (A/U/C/G/N)
                if resname not in RNA_LIKE:
                    continue

                key = (chain_id, int(resseq), str(icode if icode else " "))

                # Count residues for A/U/C/G/N
                if key not in res_key_to_idx:
                    res_key_to_idx[key] = len(res_keys)
                    res_keys.append(key)
                    if resname in BASE_TO_IDX:
                        res_base_idx.append(BASE_TO_IDX[resname])  # 0..3
                    else:
                        # 'N' placeholder
                        res_base_idx.append(-1)

                res_idx = res_key_to_idx[key]
                base_idx = res_base_idx[res_idx]  # 0..3 or -1

                # Unknown base 'N': counted in N_res but has NO pseudo-atoms
                if base_idx < 0:
                    continue

                # Keep only pseudo-atoms
                aname = _norm_atom_name(aname)
                if aname not in _KEEP_ATOMS:
                    continue

                # Atom family
                if aname == "C4'":
                    fam = 0
                elif aname == "N1":
                    fam = 1
                elif aname == "N9":
                    fam = 2
                else:  # "P"
                    fam = 3

                code = compute_code(resname, aname)
                if code < 0:
                    continue

                atoms.append((x, y, z))
                atom_codes.append(code)
                atom_res_indices.append(res_idx)
                atom_family.append(fam)

        if len(res_keys) == 0:
            raise RuntimeError(f"[RNA_Structure_Fixed] No RNA-like residues (A/U/C/G/N) found in: {pdb_path}")

        if len(atoms) == 0:
            raise RuntimeError(f"[RNA_Structure_Fixed] No valid pseudo-atoms (C4'/N1/N9/P) found in: {pdb_path}")

        self.coords = torch.tensor(np.asarray(atoms, dtype=np.float32), dtype=torch.float32, device=device)
        self.atom_codes = torch.tensor(np.asarray(atom_codes, dtype=np.int64), dtype=torch.long, device=device)
        self.atom_res_indices = torch.tensor(np.asarray(atom_res_indices, dtype=np.int64), dtype=torch.long, device=device)
        self.atom_family = torch.tensor(np.asarray(atom_family, dtype=np.int64), dtype=torch.long, device=device)

        self.N_atoms = int(self.coords.shape[0])
        self.res_keys = res_keys
        self.N_res = int(len(res_keys))

        # native_base_idx: length N_res, values in {0,1,2,3} for A/U/C/G, and -1 for N
        self.native_base_idx = torch.tensor(np.asarray(res_base_idx, dtype=np.int64), dtype=torch.long, device=device)

        # Precompute triplets + distance bins
        self._precompute_triplets(R0, bin_width)

    # ----------------- precompute i<j<k triplets and bins ----------------- #

    def _precompute_triplets(self, R0: float, bin_width: float):
        device = self.device
        coords = self.coords
        N = self.N_atoms

        # emulate C fine stage: R0_eff = R0 - 0.3 to avoid bin overflow
        R0_eff = R0 - 0.3
        if R0_eff <= 0:
            R0_eff = R0
        R0_sq = R0_eff * R0_eff

        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = (diff * diff).sum(dim=-1)

        mask_ij = (dist_sq < R0_sq) & torch.triu(
            torch.ones((N, N), dtype=torch.bool, device=device), diagonal=1
        )
        idx_i, idx_j = mask_ij.nonzero(as_tuple=True)

        triplets_list = []

        # For each (i,j), find k>j with d_ik < R0_eff and d_jk < R0_eff
        # Python loops preserve the established triplet-enumeration behavior.
        for i, j in zip(idx_i.tolist(), idx_j.tolist()):
            mask_k = (dist_sq[i] < R0_sq) & (dist_sq[j] < R0_sq)
            k_candidates = mask_k.nonzero(as_tuple=True)[0]
            for k in k_candidates.tolist():
                if k <= j:
                    continue
                triplets_list.append((i, j, k))

        if not triplets_list:
            self.triplets = torch.empty((0, 3), dtype=torch.long, device=device)
            self.b12 = torch.empty((0,), dtype=torch.long, device=device)
            self.b13 = torch.empty((0,), dtype=torch.long, device=device)
            self.b23 = torch.empty((0,), dtype=torch.long, device=device)
            self.maxbin = 0
            return

        triplets = torch.tensor(triplets_list, dtype=torch.long, device=device)
        self.triplets = triplets

        i_idx = triplets[:, 0]
        j_idx = triplets[:, 1]
        k_idx = triplets[:, 2]

        d12 = dist_sq[i_idx, j_idx].sqrt()
        d13 = dist_sq[i_idx, k_idx].sqrt()
        d23 = dist_sq[j_idx, k_idx].sqrt()

        b12 = torch.floor(d12 / bin_width).long()
        b13 = torch.floor(d13 / bin_width).long()
        b23 = torch.floor(d23 / bin_width).long()

        self.b12 = b12
        self.b13 = b13
        self.b23 = b23

        self.maxbin = int(torch.max(torch.stack([b12, b13, b23])).item())

    def get_native_sequence(self) -> str:
        """
        For unknown 'N' (native_base_idx=-1), output 'N' in sequence string.
        """
        idx = self.native_base_idx.detach().cpu().tolist()
        out = []
        for v in idx:
            if v in IDX_TO_BASE:
                out.append(IDX_TO_BASE[v])
            else:
                out.append("N")
        return "".join(out)
