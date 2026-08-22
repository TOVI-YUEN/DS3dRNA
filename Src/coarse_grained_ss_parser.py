#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import os
import math


# ============================================================
# vector helpers
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def v_add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (float(a[0] + b[0]), float(a[1] + b[1]), float(a[2] + b[2]))


def v_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (float(a[0] - b[0]), float(a[1] - b[1]), float(a[2] - b[2]))


def v_scale(a: Sequence[float], s: float) -> Tuple[float, float, float]:
    return (float(a[0] * s), float(a[1] * s), float(a[2] * s))


def v_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def v_cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        float(a[1] * b[2] - a[2] * b[1]),
        float(a[2] * b[0] - a[0] * b[2]),
        float(a[0] * b[1] - a[1] * b[0]),
    )


def v_norm(a: Sequence[float]) -> float:
    return math.sqrt(max(0.0, v_dot(a, a)))


def v_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return v_norm(v_sub(a, b))


def v_unit(a: Sequence[float], eps: float = 1e-12) -> Optional[Tuple[float, float, float]]:
    n = v_norm(a)
    if n <= eps:
        return None
    inv = 1.0 / n
    return (float(a[0] * inv), float(a[1] * inv), float(a[2] * inv))


def v_proj_perp(a: Sequence[float], axis_unit: Sequence[float]) -> Tuple[float, float, float]:
    c = v_dot(a, axis_unit)
    return v_sub(a, v_scale(axis_unit, c))


# ============================================================
# 3x3 matrix helpers (row-major)
# ============================================================

def mat_from_cols(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    return (
        (float(x[0]), float(y[0]), float(z[0])),
        (float(x[1]), float(y[1]), float(z[1])),
        (float(x[2]), float(y[2]), float(z[2])),
    )


def mat_t(m):
    return (
        (float(m[0][0]), float(m[1][0]), float(m[2][0])),
        (float(m[0][1]), float(m[1][1]), float(m[2][1])),
        (float(m[0][2]), float(m[1][2]), float(m[2][2])),
    )


def mat_mul(a, b):
    bt = mat_t(b)
    out = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(float(v_dot(a[i], bt[j])))
        out.append(tuple(row))
    return tuple(out)


def mat_trace(m) -> float:
    return float(m[0][0] + m[1][1] + m[2][2])


# ============================================================
# canonical residue mapping
# ============================================================

CANONICAL_MAP: Dict[str, str] = {
    "A": "A", "DA": "A", "ADE": "A",
    "G": "G", "DG": "G", "GUA": "G",
    "C": "C", "DC": "C", "CYT": "C",
    "U": "U", "T": "U", "DT": "U", "DU": "U", "URA": "U", "URI": "U",
    "I": "G", "DI": "G", "INO": "G",
    "1MA": "A", "2MA": "A", "M2A": "A", "6MA": "A", "M6A": "A", "A2M": "A", "RIA": "A", "12A": "A", "AVC": "A", "AMP": "A", "ATP": "A",
    "1MG": "G", "2MG": "G", "M2G": "G", "7MG": "G", "OMG": "G", "YG": "G", "G7M": "G", "QUO": "G", "GMP": "G", "GTP": "G",
    "5MC": "C", "OMC": "C", "DOC": "C", "CCC": "C", "CMP": "C",
    "PSU": "U", "H2U": "U", "5MU": "U", "OMU": "U", "4SU": "U", "S4U": "U", "MNU": "U", "DHU": "U", "UMP": "U", "TMP": "U",
    "RA": "A", "RG": "G", "RC": "C", "RU": "U",
    "ADN": "A", "GDN": "G", "CDN": "C", "TDN": "U",
}


def canonical_base(resname: str) -> str:
    return CANONICAL_MAP.get(str(resname).strip().upper(), str(resname).strip().upper())


# ============================================================
# data classes
# ============================================================

@dataclass
class ResidueCG:
    chain_id: str
    resseq: int
    icode: str
    resname_raw: str
    base: str
    p: Optional[Tuple[float, float, float]] = None
    c4: Optional[Tuple[float, float, float]] = None
    n: Optional[Tuple[float, float, float]] = None
    n_atom_name: Optional[str] = None
    index: int = -1
    fragment_id: int = 0
    frame_R: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = None
    frame_x: Optional[Tuple[float, float, float]] = None
    frame_y: Optional[Tuple[float, float, float]] = None
    frame_z: Optional[Tuple[float, float, float]] = None


@dataclass
class RoughSSConfig:
    # pairing rules
    allow_gu_wobble: bool = True
    allow_noncanonical: bool = False
    min_seq_sep: int = 4

    # Watson-Crick constraints
    wc_nn_min: float = 8.4
    wc_nn_max: float = 9.5
    wc_nn_ideal: float = 8.84

    # Non-WC constraints
    nonwc_nn_min: float = 7.5
    nonwc_nn_max: float = 13.0

    require_mutual_best: bool = True
    min_score_gap: float = 0.20

    # Topology tolerance
    allow_pseudoknot: bool = True
    cross_chain_tolerance: float = 0.6
    chain_break_tolerance: float = 0.3

    # Physical constraints
    stacking_bonus: float = 0.15
    max_c4_step_dist: float = 7.5
    keep_isolated_pairs: bool = True
    isolated_keep_threshold: float = 0.15

    # Chain break constraint
    break_c4_to_p_max: float = 8.0

    # SO(3) orientation refinement
    use_so3_refine: bool = True
    so3_gamma: float = 2.0
    so3_beta: float = 1.0
    so3_lambda_local: float = 0.90
    so3_lambda_global: float = 0.20
    so3_score_floor: float = 0.05
    so3_trigger_gap: float = 0.30
    so3_eps: float = 1e-8

    # Final selection mode.
    # original: original mutual-best/gap + greedy bracket selection.
    # maxpair : geometry-constrained maximum-pair greedy selection.
    selection_mode: str = "maxpair"

    # In maxpair mode, candidates with score above this value are not selected.
    # None or <=0 means no extra score cutoff beyond the existing geometry windows.
    maxpair_score_max: Optional[float] = None

    # Unknown-sequence scoring penalties.
    # These are not geometry-window changes; they only rank competing candidates.
    unknown_same_class_penalty: float = 0.35
    unknown_missing_class_penalty: float = 0.15

    # Two-pass pseudoknot-sandwich cleanup.
    enable_pk_sandwich_cleanup: bool = True
    pk_cleanup_iters: int = 2
    pk_sandwich_min_pk_pairs: int = 4
    pk_sandwich_max_inner_pairs: int = 2
    pk_sandwich_min_ratio: float = 2.0

    # Visibility-aware local competition.
    use_visibility_competition: bool = True
    visibility_dot_min: float = -0.10
    visibility_missing_is_visible: bool = True

    # Final island-gap-island repair.
    # After normal selection and pk-sandwich cleanup, selected runs on each
    # stacking diagonal are examined. Only island -- gap -- island is filled.
    # The gap is filled only if every missing pair is already a legal candidate.
    enable_stem_gap_fill: bool = True

    # None or <=0 means no score cutoff; any valid candidate can be filled.
    stem_gap_fill_score_max: Optional[float] = None

    # Require backbone continuity across the candidate gap. In DS3dRNA auto-SS
    # this is usually False because '&' / chain-break contexts can be valid.
    stem_gap_fill_require_continuity: bool = False

    # run_len <= this value is treated as an island; run_len > this is land.
    # Default 1 means only single-pair islands are connected.
    stem_gap_fill_flank_max_run: int = 1

    # Safety for post-selection pair repair.
    # A repaired pair is accepted only if reassigning DBN keeps all existing
    # pair associations intact. When True, existing bracket levels must also
    # remain unchanged.
    repair_require_preserve_existing_levels: bool = True
    
    
@dataclass
class PairCandidate:
    i: int
    j: int
    score: float
    nn_dist: float
    pair_type: str
    ori_quality: float = 1.0
    ori_rot_weight: float = 1.0
    ori_face_weight: float = 1.0


@dataclass
class ParseResult:
    dbn_plain: str
    dbn_with_breaks: str
    kept_pairs: List[Tuple[int, int]]
    break_after: List[bool]
    residues: List[ResidueCG]


# ============================================================
# PDB parsing
# ============================================================

def _parse_pdb_atom_line(line: str):
    record = line[0:6].strip()
    if record not in {"ATOM", "HETATM"}:
        return None

    atom_name = line[12:16].strip()
    altloc = line[16:17].strip()
    resname = line[17:20].strip()
    chain_id = line[21:22].strip()
    resseq_s = line[22:26].strip()
    icode = line[26:27].strip()

    try:
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
    except ValueError:
        return None

    if altloc not in {"", "A"}:
        return None
    if not resseq_s:
        return None

    try:
        resseq = int(resseq_s)
    except ValueError:
        return None

    return record, atom_name, resname, chain_id, resseq, icode, (x, y, z)


def load_cg_residues_from_pdb(pdb_path: str) -> List[ResidueCG]:
    """
    Load coarse-grained residues while preserving sequence length as much as possible.

    A residue is kept if at least one relevant CG atom was observed among P, C4', N1, N9.
    Missing pairing geometry later forces that position to remain unpaired rather than
    shortening the sequence/DBN.
    """
    residues: Dict[Tuple[str, int, str], ResidueCG] = {}

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = _parse_pdb_atom_line(line)
            if rec is None:
                continue

            _, atom_name, resname, chain_id, resseq, icode, xyz = rec
            atom_u = atom_name.upper().replace("*", "'")
            key = (chain_id, resseq, icode)

            if key not in residues:
                residues[key] = ResidueCG(
                    chain_id=chain_id,
                    resseq=resseq,
                    icode=icode,
                    resname_raw=resname,
                    base=canonical_base(resname),
                )

            r = residues[key]

            if atom_u == "P" and r.p is None:
                r.p = xyz
            elif atom_u in {"C4'", "C4’"} and r.c4 is None:
                r.c4 = xyz
            elif atom_u == "N1":
                if r.n is None or r.n_atom_name != "N9":
                    r.n = xyz
                    r.n_atom_name = "N1"
            elif atom_u == "N9":
                if r.n is None or r.n_atom_name != "N1":
                    r.n = xyz
                    r.n_atom_name = "N9"

    out: List[ResidueCG] = []
    for key in sorted(residues.keys(), key=lambda t: (t[0], t[1], t[2])):
        r = residues[key]
        if r.p is None and r.c4 is None and r.n is None:
            continue
        out.append(r)

    for i, r in enumerate(out):
        r.index = i

    return out


# ============================================================
# chain break inference & fragmentation
# ============================================================

def assign_breaks_and_fragments(residues: Sequence[ResidueCG], cfg: RoughSSConfig) -> List[bool]:
    n = len(residues)
    if n == 0:
        return []

    break_after = [False] * n
    curr_frag = 0

    for i in range(n - 1):
        residues[i].fragment_id = curr_frag
        a = residues[i]
        b = residues[i + 1]

        is_break = False
        if a.chain_id != b.chain_id:
            is_break = True
        elif a.c4 is not None and b.p is not None:
            if v_dist(a.c4, b.p) > cfg.break_c4_to_p_max:
                is_break = True

        if is_break:
            break_after[i] = True
            curr_frag += 1

    if n > 0:
        residues[n - 1].fragment_id = curr_frag

    return break_after


# ============================================================
# SO(3) local-frame construction
# ============================================================

def _same_fragment(a: ResidueCG, b: ResidueCG) -> bool:
    return a.chain_id == b.chain_id and a.fragment_id == b.fragment_id


def _choose_backbone_vector(residues: Sequence[ResidueCG], idx: int) -> Optional[Tuple[float, float, float]]:
    r = residues[idx]
    if r.c4 is None:
        return None

    candidates: List[Tuple[float, float, float]] = []

    if r.p is not None:
        candidates.append(v_sub(r.p, r.c4))

    if idx + 1 < len(residues) and residues[idx + 1].c4 is not None and _same_fragment(r, residues[idx + 1]):
        candidates.append(v_sub(residues[idx + 1].c4, r.c4))

    if idx - 1 >= 0 and residues[idx - 1].c4 is not None and _same_fragment(residues[idx - 1], r):
        candidates.append(v_sub(r.c4, residues[idx - 1].c4))

    if idx - 1 >= 0 and idx + 1 < len(residues):
        rp = residues[idx - 1]
        rn = residues[idx + 1]
        if rp.c4 is not None and rn.c4 is not None and _same_fragment(rp, r) and _same_fragment(r, rn):
            candidates.append(v_sub(rn.c4, rp.c4))

    for vec in candidates:
        u = v_unit(vec)
        if u is not None:
            return u
    return None


def build_local_frames(residues: Sequence[ResidueCG]) -> None:
    for idx, r in enumerate(residues):
        r.frame_R = None
        r.frame_x = None
        r.frame_y = None
        r.frame_z = None

        if r.c4 is None or r.n is None:
            continue

        x_axis = v_unit(v_sub(r.n, r.c4))
        if x_axis is None:
            continue

        y_seed = _choose_backbone_vector(residues, idx)
        if y_seed is None:
            continue

        y_perp = v_proj_perp(y_seed, x_axis)
        y_axis = v_unit(y_perp)
        if y_axis is None:
            continue

        z_axis = v_unit(v_cross(x_axis, y_axis))
        if z_axis is None:
            continue

        y_axis_2 = v_unit(v_cross(z_axis, x_axis))
        if y_axis_2 is None:
            continue

        r.frame_x = x_axis
        r.frame_y = y_axis_2
        r.frame_z = z_axis
        r.frame_R = mat_from_cols(x_axis, y_axis_2, z_axis)


# ============================================================
# pairing identity + score
# ============================================================

def is_backbone_continuous(r1: ResidueCG, r2: ResidueCG, max_dist: float) -> bool:
    if r1.c4 is None or r2.c4 is None:
        return False
    return v_dist(r1.c4, r2.c4) <= max_dist


def _base_class_from_n_atom(r: ResidueCG) -> str:
    """
    N9 -> R, purine-like, designable as A/G.
    N1 -> Y, pyrimidine-like, designable as U/C.
    """
    name = str(r.n_atom_name or "").strip().upper()
    if name == "N9":
        return "R"
    if name == "N1":
        return "Y"
    return "N"


def _nonwc_score_from_distance(d: float, lo: float, hi: float) -> float:
    mid = (hi + lo) / 2.0
    flat_radius = (hi - lo) * 0.35
    deviation = max(0.0, abs(d - mid) - flat_radius)
    return deviation + 0.5


def candidate_score(a: ResidueCG, b: ResidueCG, cfg: RoughSSConfig) -> Optional[PairCandidate]:

    if a.c4 is None or a.n is None or b.c4 is None or b.n is None:
        return None

    ba = canonical_base(a.base)
    bb = canonical_base(b.base)
    d = v_dist(a.n, b.n)

    is_cross_chain = (a.chain_id != b.chain_id)
    is_chain_break = (a.chain_id == b.chain_id) and (a.fragment_id != b.fragment_id)

    tol = 0.0
    if is_cross_chain:
        tol = cfg.cross_chain_tolerance
    elif is_chain_break:
        tol = cfg.chain_break_tolerance

    wc_max = cfg.wc_nn_max + tol
    nwc_max = cfg.nonwc_nn_max + tol

    known_bases = {"A", "U", "C", "G"}
    is_unknown = (ba not in known_bases) or (bb not in known_bases)

    # ========================================================
    # Unknown / design-time mode
    # ========================================================
    if is_unknown:
        ca = _base_class_from_n_atom(a)
        cb = _base_class_from_n_atom(b)

        mixed_RY = (ca == "R" and cb == "Y") or (ca == "Y" and cb == "R")
        same_class = (ca in {"R", "Y"} and cb in {"R", "Y"} and ca == cb)

        best: Optional[PairCandidate] = None

        # Canonical WC-like geometry only makes sense for purine-pyrimidine class.
        if mixed_RY and cfg.wc_nn_min <= d <= wc_max:
            score_wc = abs(d - cfg.wc_nn_ideal)
            best = PairCandidate(
                i=a.index,
                j=b.index,
                score=score_wc,
                nn_dist=d,
                pair_type="WC_like_RY",
            )

        # nonWC-like fallback for distorted R-Y, same-class tertiary-like contacts,
        # or missing/ambiguous N-class.
        if cfg.nonwc_nn_min <= d <= nwc_max:
            score_nonwc = _nonwc_score_from_distance(d, cfg.nonwc_nn_min, nwc_max)

            if same_class:
                score_nonwc += float(cfg.unknown_same_class_penalty)
                ptype = "nonWC_like_same_class"
            elif mixed_RY:
                ptype = "nonWC_like_RY"
            else:
                score_nonwc += float(cfg.unknown_missing_class_penalty)
                ptype = "nonWC_like_unknown_class"

            cand_nonwc = PairCandidate(
                i=a.index,
                j=b.index,
                score=score_nonwc,
                nn_dist=d,
                pair_type=ptype,
            )

            if best is None or cand_nonwc.score < best.score:
                best = cand_nonwc

        return best

    # ========================================================
    # Known-base mode: original behavior
    # ========================================================
    s = {ba, bb}
    is_wc = (s == {"A", "U"} or s == {"G", "C"})
    is_gu = (s == {"G", "U"})

    if is_wc:
        if cfg.wc_nn_min <= d <= wc_max:
            score = abs(d - cfg.wc_nn_ideal)
            return PairCandidate(i=a.index, j=b.index, score=score, nn_dist=d, pair_type="WC")

    elif (cfg.allow_gu_wobble and is_gu) or cfg.allow_noncanonical:
        if cfg.nonwc_nn_min <= d <= nwc_max:
            score = _nonwc_score_from_distance(d, cfg.nonwc_nn_min, nwc_max)
            ptype = "GU" if is_gu else "nonWC"
            return PairCandidate(i=a.index, j=b.index, score=score, nn_dist=d, pair_type=ptype)

    return None


def crosses(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    i, j = p1
    k, l = p2
    return (i < k < j < l) or (k < i < l < j)


# ============================================================
# SO(3) orientation scoring and conservative reweight
# ============================================================

def head_to_head_reference_rotation():
    return (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
    )


def orientation_quality(a: ResidueCG, b: ResidueCG, cfg: RoughSSConfig) -> Tuple[float, float, float]:
    if a.frame_R is None or b.frame_R is None or a.frame_x is None or b.frame_x is None:
        return 1.0, 1.0, 1.0
    if a.c4 is None or b.c4 is None:
        return 1.0, 1.0, 1.0

    r_rel = mat_mul(mat_t(a.frame_R), b.frame_R)
    h_ref = head_to_head_reference_rotation()
    q = mat_mul(mat_t(h_ref), r_rel)

    trq = clamp(mat_trace(q), -1.0, 3.0)
    cos_theta_err = clamp((trq - 1.0) / 2.0, -1.0, 1.0)
    w_rot = ((1.0 + cos_theta_err) * 0.5) ** max(cfg.so3_gamma, 0.0)

    u_ij = v_unit(v_sub(b.c4, a.c4))
    if u_ij is None:
        return w_rot, w_rot, 1.0

    fi = max(0.0, v_dot(a.frame_x, u_ij))
    fj = max(0.0, v_dot(b.frame_x, v_scale(u_ij, -1.0)))
    w_face = max(0.0, fi * fj) ** max(cfg.so3_beta, 0.0)

    return w_rot * w_face, w_rot, w_face


def _residue_can_see_partner(
    residues: Sequence[ResidueCG],
    i: int,
    j: int,
    cfg: RoughSSConfig,
) -> bool:
    """
    Visibility test for local competition only.

    This does NOT delete a candidate pair. It only decides whether candidate
    (i,j) should be allowed to compete for residue i's best/second-best partner.

    C4'->N is used as a weak base-facing proxy:
        visible if dot(frame_x_i, unit(C4_i -> C4_j)) >= threshold
    """
    if not getattr(cfg, "use_visibility_competition", True):
        return True

    ri = residues[i]
    rj = residues[j]

    if ri.c4 is None or rj.c4 is None:
        return bool(getattr(cfg, "visibility_missing_is_visible", True))

    if ri.frame_x is None:
        build_local_frames(residues)

    if ri.frame_x is None:
        return bool(getattr(cfg, "visibility_missing_is_visible", True))

    u = v_unit(v_sub(rj.c4, ri.c4))
    if u is None:
        return bool(getattr(cfg, "visibility_missing_is_visible", True))

    dotv = v_dot(ri.frame_x, u)
    return dotv >= float(getattr(cfg, "visibility_dot_min", -0.10))


def _pair_visible_for_competition(
    residues: Sequence[ResidueCG],
    i: int,
    j: int,
    cfg: RoughSSConfig,
) -> Tuple[bool, bool]:
    """Return directional visibility flags for local competition."""
    return (
        _residue_can_see_partner(residues, i, j, cfg),
        _residue_can_see_partner(residues, j, i, cfg),
    )


def project_to_simplex(values: Sequence[float], target_sum: float) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    if target_sum <= 0.0:
        return [0.0] * n

    u = sorted((float(v) for v in values), reverse=True)
    cssv = 0.0
    rho = -1
    theta = 0.0
    for i, ui in enumerate(u, start=1):
        cssv += ui
        t = (cssv - target_sum) / i
        if ui - t > 0.0:
            rho = i
            theta = t
    if rho == -1:
        theta = (sum(u) - target_sum) / max(1, n)
    out = [max(float(v) - theta, 0.0) for v in values]

    s = sum(out)
    if s > 0.0:
        scale = target_sum / s
        out = [v * scale for v in out]
    else:
        out = [target_sum / n] * n
    return out


def apply_so3_conservative_reweight(candidates: Dict[Tuple[int, int], PairCandidate], residues: Sequence[ResidueCG], cfg: RoughSSConfig) -> None:
    if not cfg.use_so3_refine or not candidates:
        return

    build_local_frames(residues)

    keys = sorted(candidates.keys())
    neigh_scores: Dict[int, List[Tuple[Tuple[int, int], float]]] = {i: [] for i in range(len(residues))}
    neigh_qual: Dict[int, List[Tuple[Tuple[int, int], float]]] = {i: [] for i in range(len(residues))}

    qualities: Dict[Tuple[int, int], float] = {}
    for key in keys:
        cand = candidates[key]
        q, w_rot, w_face = orientation_quality(residues[cand.i], residues[cand.j], cfg)
        cand.ori_quality = q
        cand.ori_rot_weight = w_rot
        cand.ori_face_weight = w_face
        qualities[key] = q
        neigh_scores[cand.i].append((key, cand.score))
        neigh_scores[cand.j].append((key, cand.score))
        neigh_qual[cand.i].append((key, q))
        neigh_qual[cand.j].append((key, q))

    if not qualities:
        return

    q_global = sum(qualities.values()) / len(qualities)

    local_mu: Dict[int, float] = {}
    local_gate: Dict[int, float] = {}
    for i in range(len(residues)):
        arr_q = neigh_qual[i]
        local_mu[i] = sum(q for _, q in arr_q) / len(arr_q) if arr_q else q_global

        arr_s = sorted((s for _, s in neigh_scores[i]))
        if len(arr_s) < 2:
            local_gate[i] = 0.0
        else:
            gap = arr_s[1] - arr_s[0]
            if cfg.so3_trigger_gap <= 0.0:
                local_gate[i] = 1.0
            else:
                local_gate[i] = clamp(1.0 - gap / cfg.so3_trigger_gap, 0.0, 1.0)

    s_orig = [max(0.0, float(candidates[k].score)) for k in keys]
    total_orig = sum(s_orig)
    if total_orig <= cfg.so3_eps:
        return

    floor = max(0.0, cfg.so3_score_floor)
    shifted = [s + floor for s in s_orig]

    raw_adjusted_minus_floor: List[float] = []
    for key, s_bar in zip(keys, shifted):
        cand = candidates[key]
        h = qualities[key]

        di = local_gate.get(cand.i, 0.0) * (h - local_mu.get(cand.i, q_global))
        dj = local_gate.get(cand.j, 0.0) * (h - local_mu.get(cand.j, q_global))
        centered = 0.5 * (di + dj)
        global_term = h - q_global

        exponent = -(cfg.so3_lambda_local * centered + cfg.so3_lambda_global * global_term)
        m_raw = math.exp(exponent)
        raw_adjusted_minus_floor.append(s_bar * m_raw - floor)

    reweighted = project_to_simplex(raw_adjusted_minus_floor, total_orig)

    for key, new_score in zip(keys, reweighted):
        candidates[key].score = float(new_score)


# ============================================================
# candidate construction and selection
# ============================================================

def _generate_scored_candidates(residues: Sequence[ResidueCG], cfg: RoughSSConfig) -> Dict[Tuple[int, int], PairCandidate]:
    n = len(residues)
    candidates: Dict[Tuple[int, int], PairCandidate] = {}

    # 1. Generate raw candidates.
    for i in range(n):
        for j in range(i + 1, n):
            if residues[i].fragment_id == residues[j].fragment_id and abs(i - j) < cfg.min_seq_sep:
                continue

            cand = candidate_score(residues[i], residues[j], cfg)
            if cand is not None:
                candidates[(i, j)] = cand

    # 2. Strict geometric stacking bonus.
    for (i, j), cand in list(candidates.items()):
        bonus_applied = 0.0

        if (i + 1, j - 1) in candidates:
            if is_backbone_continuous(residues[i], residues[i + 1], cfg.max_c4_step_dist) and \
               is_backbone_continuous(residues[j], residues[j - 1], cfg.max_c4_step_dist):
                bonus_applied += cfg.stacking_bonus

        if (i - 1, j + 1) in candidates:
            if is_backbone_continuous(residues[i - 1], residues[i], cfg.max_c4_step_dist) and \
               is_backbone_continuous(residues[j], residues[j + 1], cfg.max_c4_step_dist):
                bonus_applied += cfg.stacking_bonus

        cand.score = max(0.0, cand.score - bonus_applied)

    # 3. SO(3)-aware conservative score reweighting.
    apply_so3_conservative_reweight(candidates, residues, cfg)
    return candidates


def _assign_pairs_to_dbn(
    n: int,
    kept_pairs: Sequence[Tuple[int, int]],
    allow_pseudoknot: bool,
) -> Tuple[str, Dict[Tuple[int, int], int]]:
    dbn = ["."] * n
    bracket_symbols = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
    bracket_levels: List[List[Tuple[int, int]]] = [[] for _ in range(len(bracket_symbols))]
    pair_to_level: Dict[Tuple[int, int], int] = {}

    for i, j in sorted(kept_pairs):
        placed_level = -1
        for lvl_idx in range(len(bracket_levels)):
            if not any(crosses((i, j), (k, l)) for k, l in bracket_levels[lvl_idx]):
                placed_level = lvl_idx
                break

        if placed_level == -1:
            continue
        if placed_level > 0 and not allow_pseudoknot:
            continue

        bracket_levels[placed_level].append((i, j))
        pair_to_level[(i, j)] = placed_level
        op, cl = bracket_symbols[placed_level]
        dbn[i] = op
        dbn[j] = cl

    return "".join(dbn), pair_to_level


def _addition_preserves_existing_pairing(
    n: int,
    existing_pairs: Sequence[Tuple[int, int]],
    add_pairs: Sequence[Tuple[int, int]],
    allow_pseudoknot: bool,
    require_same_level: bool = True,
) -> bool:
    """
    Safety check for post-selection repair steps.

    The repair is allowed only if adding ``add_pairs`` does not disturb any
    already accepted pair. This is stricter than checking endpoint occupancy:
      1. every previous pair must still be assignable after DBN reassignment;
      2. no previous pair can lose/change its partner;
      3. optionally, previous pairs must keep their original bracket level.

    This prevents a repair pair from changing the topology/level assignment of
    completed stems or pseudoknot layers.
    """
    existing_norm = sorted({tuple(sorted(p)) for p in existing_pairs})
    add_norm = sorted({tuple(sorted(p)) for p in add_pairs})

    if not add_norm:
        return True

    used = set()
    for a, b in existing_norm:
        if a >= b:
            return False
        if a in used or b in used:
            return False
        used.add(a)
        used.add(b)

    add_used = set()
    for a, b in add_norm:
        if a >= b:
            return False
        if a in used or b in used:
            return False
        if a in add_used or b in add_used:
            return False
        add_used.add(a)
        add_used.add(b)

    _old_dbn, old_level = _assign_pairs_to_dbn(
        n=n,
        kept_pairs=existing_norm,
        allow_pseudoknot=allow_pseudoknot,
    )
    _new_dbn, new_level = _assign_pairs_to_dbn(
        n=n,
        kept_pairs=existing_norm + add_norm,
        allow_pseudoknot=allow_pseudoknot,
    )

    for pair in existing_norm:
        if pair not in old_level:
            return False
        if pair not in new_level:
            return False
        if require_same_level and new_level[pair] != old_level[pair]:
            return False

    for pair in add_norm:
        if pair not in new_level:
            return False

    return True


def _select_pairs_original(
    residues: Sequence[ResidueCG],
    candidates: Dict[Tuple[int, int], PairCandidate],
    cfg: RoughSSConfig,
) -> List[Tuple[int, int]]:
    """
    Original local competition selector, upgraded with directional visibility.

    Candidate generation is unchanged. A candidate that a residue cannot "see"
    through its C4'->N proxy does not participate in that residue's best/second-best
    local competition.
    """
    n = len(residues)

    if getattr(cfg, "use_visibility_competition", True):
        build_local_frames(residues)

    neigh: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
    pair_vis: Dict[Tuple[int, int], Tuple[bool, bool]] = {}

    for (i, j), cand in candidates.items():
        vis_i, vis_j = _pair_visible_for_competition(residues, i, j, cfg)
        pair_vis[(i, j)] = (vis_i, vis_j)

        if vis_i:
            neigh[i].append((j, cand.score))
        if vis_j:
            neigh[j].append((i, cand.score))

    best_for: Dict[int, Tuple[int, float]] = {}
    second_for: Dict[int, Tuple[int, float]] = {}

    for i in range(n):
        arr = sorted(neigh[i], key=lambda x: (x[1], x[0]))
        if arr:
            best_for[i] = arr[0]
        if len(arr) > 1:
            second_for[i] = arr[1]

    preaccepted: List[Tuple[int, int, float]] = []

    for (i, j), cand in candidates.items():
        keep = True

        vis_i, vis_j = pair_vis.get((i, j), (True, True))

        if not (vis_i and vis_j):
            keep = False

        if keep:
            if i not in best_for or best_for[i][0] != j:
                keep = False

        if keep and cfg.require_mutual_best:
            if j not in best_for or best_for[j][0] != i:
                keep = False

        if keep:
            gap_i = second_for[i][1] - best_for[i][1] if i in second_for else float("inf")
            gap_j = second_for[j][1] - best_for[j][1] if j in second_for else float("inf")

            gap_threshold = cfg.min_score_gap
            if residues[i].chain_id != residues[j].chain_id:
                gap_threshold *= 0.4
            elif residues[i].fragment_id != residues[j].fragment_id:
                gap_threshold *= 0.7

            if gap_i < gap_threshold or gap_j < gap_threshold:
                keep = False

        if keep:
            preaccepted.append((i, j, cand.score))

    preaccepted.sort(key=lambda x: (x[2], x[0], x[1]))

    used = set()
    kept_pairs: List[Tuple[int, int]] = []
    bracket_levels: List[List[Tuple[int, int]]] = [[] for _ in range(4)]

    for i, j, _score in preaccepted:
        if i in used or j in used:
            continue

        placed_level = -1
        for lvl_idx in range(len(bracket_levels)):
            if not any(crosses((i, j), (k, l)) for k, l in bracket_levels[lvl_idx]):
                placed_level = lvl_idx
                break

        if placed_level == -1:
            continue
        if placed_level > 0 and not cfg.allow_pseudoknot:
            continue

        bracket_levels[placed_level].append((i, j))
        kept_pairs.append((i, j))
        used.add(i)
        used.add(j)

    if not cfg.keep_isolated_pairs:
        pair_set = set(kept_pairs)
        filtered_pairs = []
        for i, j in kept_pairs:
            stacked = (i + 1, j - 1) in pair_set or (i - 1, j + 1) in pair_set
            keep_anyway = False

            if not stacked:
                cand = candidates[(i, j)]
                is_cross = (residues[i].chain_id != residues[j].chain_id)

                if cand.pair_type in {"WC", "WC_like_RY"} and cand.score <= cfg.isolated_keep_threshold:
                    keep_anyway = True
                elif is_cross and cand.score <= (cfg.isolated_keep_threshold * 1.5):
                    keep_anyway = True

            if stacked or keep_anyway:
                filtered_pairs.append((i, j))

        kept_pairs = filtered_pairs

    kept_pairs.sort()
    return kept_pairs

def _select_pairs_maxpair(
    residues: Sequence[ResidueCG],
    candidates: Dict[Tuple[int, int], PairCandidate],
    cfg: RoughSSConfig,
) -> List[Tuple[int, int]]:
    """
    Geometry-constrained maximum-pair greedy selector.

    This does not change candidate geometry windows. It only changes the final
    selection objective: keep as many valid pairs as possible, with lower score
    as tie-break. It supports pseudoknot levels using the same bracket logic as
    the original parser.
    """
    score_cut = cfg.maxpair_score_max

    items: List[Tuple[int, int, float]] = []
    for (i, j), cand in candidates.items():
        if score_cut is not None and float(score_cut) > 0 and cand.score > float(score_cut):
            continue
        items.append((i, j, cand.score))

    # Lower score first, then prefer stacked-like local neighbors implicitly by
    # score after stacking/SO3 reweight. This keeps quality as tie-break while
    # not imposing mutual-best/gap bottlenecks.
    items.sort(key=lambda x: (x[2], x[0], x[1]))

    used = set()
    kept_pairs: List[Tuple[int, int]] = []
    bracket_levels: List[List[Tuple[int, int]]] = [[] for _ in range(4)]

    for i, j, _score in items:
        if i in used or j in used:
            continue

        placed_level = -1
        for lvl_idx in range(len(bracket_levels)):
            if not any(crosses((i, j), (k, l)) for k, l in bracket_levels[lvl_idx]):
                placed_level = lvl_idx
                break

        if placed_level == -1:
            continue
        if placed_level > 0 and not cfg.allow_pseudoknot:
            continue

        bracket_levels[placed_level].append((i, j))
        kept_pairs.append((i, j))
        used.add(i)
        used.add(j)

    if not cfg.keep_isolated_pairs:
        pair_set = set(kept_pairs)
        filtered_pairs = []
        for i, j in kept_pairs:
            stacked = (i + 1, j - 1) in pair_set or (i - 1, j + 1) in pair_set
            keep_anyway = False
            cand = candidates[(i, j)]
            is_cross = residues[i].chain_id != residues[j].chain_id

            if stacked:
                keep_anyway = True
            elif cand.pair_type in {"WC", "WC_like_RY"} and cand.score <= cfg.isolated_keep_threshold:
                keep_anyway = True
            elif is_cross and cand.score <= (cfg.isolated_keep_threshold * 1.5):
                keep_anyway = True

            if keep_anyway:
                filtered_pairs.append((i, j))

        kept_pairs = filtered_pairs

    kept_pairs.sort()
    return kept_pairs


# ============================================================
# main parser
# ============================================================

def _filter_candidates_forced_unpaired(
    candidates: Dict[Tuple[int, int], PairCandidate],
    forced_unpaired: set,
) -> Dict[Tuple[int, int], PairCandidate]:
    if not forced_unpaired:
        return candidates
    return {
        (i, j): cand
        for (i, j), cand in candidates.items()
        if i not in forced_unpaired and j not in forced_unpaired
    }


def _pair_stack_groups(pairs: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    if not pairs:
        return []

    pairs = sorted(pairs, key=lambda x: (x[0], -x[1]))
    groups: List[List[Tuple[int, int]]] = []

    for p in pairs:
        placed = False
        for g in groups:
            last = g[-1]
            if p[0] == last[0] + 1 and p[1] == last[1] - 1:
                g.append(p)
                placed = True
                break
        if not placed:
            groups.append([p])

    return groups


def _detect_pk_sandwich_forced_unpaired(
    kept_pairs: List[Tuple[int, int]],
    pair_to_level: Dict[Tuple[int, int], int],
    cfg: RoughSSConfig,
) -> set:
    forced = set()
    if not getattr(cfg, "enable_pk_sandwich_cleanup", True):
        return forced

    min_pk = int(getattr(cfg, "pk_sandwich_min_pk_pairs", 4))
    max_inner = int(getattr(cfg, "pk_sandwich_max_inner_pairs", 2))
    min_ratio = float(getattr(cfg, "pk_sandwich_min_ratio", 2.0))

    if min_pk <= 0:
        return forced

    levels = sorted(set(pair_to_level.get(p, 0) for p in kept_pairs))
    for lvl in levels:
        if lvl <= 0:
            continue

        pk_pairs = [p for p in kept_pairs if pair_to_level.get(p, 0) == lvl]
        for group in _pair_stack_groups(pk_pairs):
            if len(group) < min_pk:
                continue

            left_min = min(i for i, _j in group)
            right_max = max(j for _i, j in group)

            inner_pairs = []
            for p in kept_pairs:
                if p in group:
                    continue

                a, b = p
                plvl = pair_to_level.get(p, 0)

                if left_min < a < b < right_max and plvl != lvl:
                    inner_pairs.append(p)

            if not inner_pairs:
                continue
            if len(inner_pairs) > max_inner:
                continue
            if len(group) < min_ratio * max(1, len(inner_pairs)):
                continue

            for a, b in inner_pairs:
                forced.add(a)
                forced.add(b)

    return forced


def _select_and_assign_once(
    n: int,
    residues: Sequence[ResidueCG],
    candidates: Dict[Tuple[int, int], PairCandidate],
    cfg: RoughSSConfig,
) -> Tuple[List[Tuple[int, int]], str, Dict[Tuple[int, int], int]]:
    """
    One normal parsing pass only.

    Important:
      Gap repair must NOT run here. It is a final post-processing step after
      all pk-sandwich cleanup iterations have finished. Keeping this function
      clean prevents gap repair from being undone or distorted by the next
      forced-unpair iteration.
    """
    mode = str(getattr(cfg, "selection_mode", "maxpair")).strip().lower()

    if mode == "original":
        kept_pairs = _select_pairs_original(residues, candidates, cfg)
    else:
        kept_pairs = _select_pairs_maxpair(residues, candidates, cfg)

    dbn_plain, pair_to_level = _assign_pairs_to_dbn(
        n=n,
        kept_pairs=kept_pairs,
        allow_pseudoknot=cfg.allow_pseudoknot,
    )

    kept_pairs = sorted([p for p in kept_pairs if p in pair_to_level])

    return kept_pairs, dbn_plain, pair_to_level

def _stack_diagonal_key(pair: Tuple[int, int]) -> int:
    """
    Stacked pairs (i,j), (i+1,j-1), ... share i+j.
    """
    i, j = pair
    return int(i + j)


def _selected_runs_on_diagonal(
    selected_pairs: List[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """
    Group selected pairs on the same stacking diagonal into consecutive runs.
    Consecutive means:
        (i,j), (i+1,j-1), (i+2,j-2), ...
    """
    if not selected_pairs:
        return []

    pairs = sorted(selected_pairs, key=lambda p: p[0])
    runs: List[List[Tuple[int, int]]] = []

    cur = [pairs[0]]
    for p in pairs[1:]:
        last = cur[-1]
        if p[0] == last[0] + 1 and p[1] == last[1] - 1:
            cur.append(p)
        else:
            runs.append(cur)
            cur = [p]

    runs.append(cur)
    return runs


def _gap_pairs_between_runs(
    left_run: List[Tuple[int, int]],
    right_run: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Return missing stacked positions between two selected runs on same diagonal.

    Example:
        left_run  ends at (i,j)
        right_run starts at (i+k,j-k)

    gap = [(i+1,j-1), ..., (i+k-1,j-k+1)]
    """
    if not left_run or not right_run:
        return []

    a0, b0 = left_run[-1]
    a1, b1 = right_run[0]

    gap: List[Tuple[int, int]] = []
    a, b = a0 + 1, b0 - 1

    while a < a1 and b > b1:
        gap.append((a, b))
        a += 1
        b -= 1

    return gap


def _gap_has_backbone_continuity(
    residues: Sequence[ResidueCG],
    gap_pairs: List[Tuple[int, int]],
    cfg: RoughSSConfig,
) -> bool:
    """
    Check local continuity along both strands across the gap region.

    This is deliberately conservative: if required atoms are missing,
    continuity fails only when is_backbone_continuous() cannot confirm it.
    """
    if not gap_pairs:
        return False

    max_d = float(cfg.max_c4_step_dist)

    for a, b in gap_pairs:
        n = len(residues)

        if a - 1 >= 0:
            if not is_backbone_continuous(residues[a - 1], residues[a], max_d):
                return False
        if a + 1 < n:
            if not is_backbone_continuous(residues[a], residues[a + 1], max_d):
                return False

        if b + 1 < n:
            if not is_backbone_continuous(residues[b + 1], residues[b], max_d):
                return False
        if b - 1 >= 0:
            if not is_backbone_continuous(residues[b], residues[b - 1], max_d):
                return False

    return True

def _fill_single_gap_pairs_by_stem_context(
    residues: Sequence[ResidueCG],
    kept_pairs: List[Tuple[int, int]],
    candidates: Dict[Tuple[int, int], PairCandidate],
    cfg: RoughSSConfig,
) -> List[Tuple[int, int]]:

    if not getattr(cfg, "enable_stem_gap_fill", True):
        return kept_pairs

    if not kept_pairs or not candidates:
        return kept_pairs

    score_cut = getattr(cfg, "stem_gap_fill_score_max", None)
    require_cont = bool(getattr(cfg, "stem_gap_fill_require_continuity", False))
    island_max_run = int(getattr(cfg, "stem_gap_fill_flank_max_run", 1))
    require_same_level = bool(
        getattr(cfg, "repair_require_preserve_existing_levels", True)
    )

    pair_set = set(tuple(sorted(p)) for p in kept_pairs)

    def _pair_score_sum(ps: List[Tuple[int, int]]) -> float:
        return float(sum(candidates[p].score for p in ps if p in candidates))

    def _passes_add_checks(
        add_pairs: List[Tuple[int, int]],
        partner_of: Dict[int, int],
    ) -> bool:
        if not add_pairs:
            return False

        for gp in add_pairs:
            if gp in pair_set:
                return False

            if gp not in candidates:
                return False

            a, b = gp
            if a >= b:
                return False

            if a in partner_of or b in partner_of:
                return False

            cand = candidates[gp]
            if score_cut is not None and float(score_cut) > 0.0:
                if cand.score > float(score_cut):
                    return False

        if require_cont:
            if not _gap_has_backbone_continuity(residues, add_pairs, cfg):
                return False

        if not _addition_preserves_existing_pairing(
            n=len(residues),
            existing_pairs=sorted(pair_set),
            add_pairs=add_pairs,
            allow_pseudoknot=cfg.allow_pseudoknot,
            require_same_level=require_same_level,
        ):
            return False

        return True

    def _next_position_blocked_after(
        p: Tuple[int, int],
        direction: int,
        partner_of: Dict[int, int],
    ) -> bool:
        """
        direction = +1: next is inward  (i+1, j-1)
        direction = -1: next is outward (i-1, j+1)

        Blocked means:
          - outside valid order;
          - already selected;
          - not a candidate;
          - or endpoint already paired.

        This prevents passive repair from becoming general extension.
        """
        a, b = p
        q = (a + direction, b - direction)

        if q[0] >= q[1]:
            return True

        if q in pair_set:
            return True

        if q not in candidates:
            return True

        if q[0] in partner_of or q[1] in partner_of:
            return True

        return False

    changed = True
    while changed:
        changed = False

        by_diag: Dict[int, List[Tuple[int, int]]] = {}
        for p in pair_set:
            by_diag.setdefault(_stack_diagonal_key(p), []).append(p)

        partner_of: Dict[int, int] = {}
        for i, j in pair_set:
            partner_of[i] = j
            partner_of[j] = i

        # proposal:
        #   (priority, gap_len, score_sum, diag, anchor0, anchor1, add_pairs)
        # priority 0 = active island-gap-island
        # priority 1 = passive terminal island repair
        proposals: List[
            Tuple[int, int, float, int, int, int, List[Tuple[int, int]]]
        ] = []

        for diag, pairs_on_diag in by_diag.items():
            runs = _selected_runs_on_diagonal(pairs_on_diag)
            if len(runs) < 2:
                continue

            # ============================================================
            # 1. Active repair: adjacent island -- gap -- island ONLY.
            # ============================================================
            for li in range(len(runs) - 1):
                left_run = runs[li]
                right_run = runs[li + 1]

                if len(left_run) > island_max_run:
                    continue
                if len(right_run) > island_max_run:
                    continue

                gap_pairs = _gap_pairs_between_runs(left_run, right_run)

                # Conservative cap: only tiny holes between adjacent islands.
                if not gap_pairs or len(gap_pairs) > 2:
                    continue

                # Anti-leapfrog: adjacent runs should not contain selected pairs
                # in their gap; keep this explicit for safety.
                if any(gp in pair_set for gp in gap_pairs):
                    continue

                if not _passes_add_checks(gap_pairs, partner_of):
                    continue

                proposals.append(
                    (
                        0,
                        len(gap_pairs),             # shorter first
                        _pair_score_sum(gap_pairs),
                        int(diag),
                        int(left_run[0][0]),
                        int(right_run[0][0]),
                        gap_pairs,
                    )
                )

            # ============================================================
            # 2. Passive terminal island repair:
            #
            #    mainland -- gap -- island
            #
            #    Do NOT bridge toward mainland.
            #    Instead extend the island by exactly one pair on the opposite
            #    side, and only if the next position is blocked.
            # ============================================================
            for idx, run in enumerate(runs):
                if len(run) > island_max_run:
                    continue  # not an island

                # Case A:
                # left neighbor is mainland, island extends inward/right.
                if idx - 1 >= 0:
                    left_run = runs[idx - 1]
                    if len(left_run) > island_max_run:
                        bridge_gap = _gap_pairs_between_runs(left_run, run)

                        # Only local mainland-gap-island contexts.
                        if bridge_gap and len(bridge_gap) <= 2:
                            last_i, last_j = run[-1]
                            p = (last_i + 1, last_j - 1)

                            if p[0] < p[1]:
                                if (
                                    p not in pair_set
                                    and p in candidates
                                    and p[0] not in partner_of
                                    and p[1] not in partner_of
                                    and _next_position_blocked_after(p, +1, partner_of)
                                    and _passes_add_checks([p], partner_of)
                                ):
                                    proposals.append(
                                        (
                                            1,
                                            1,
                                            _pair_score_sum([p]),
                                            int(diag),
                                            int(run[0][0]),
                                            int(run[-1][0]),
                                            [p],
                                        )
                                    )

                # Case B:
                # right neighbor is mainland, island extends outward/left.
                if idx + 1 < len(runs):
                    right_run = runs[idx + 1]
                    if len(right_run) > island_max_run:
                        bridge_gap = _gap_pairs_between_runs(run, right_run)

                        # Only local island-gap-mainland contexts.
                        if bridge_gap and len(bridge_gap) <= 2:
                            first_i, first_j = run[0]
                            p = (first_i - 1, first_j + 1)

                            if p[0] < p[1]:
                                if (
                                    p not in pair_set
                                    and p in candidates
                                    and p[0] not in partner_of
                                    and p[1] not in partner_of
                                    and _next_position_blocked_after(p, -1, partner_of)
                                    and _passes_add_checks([p], partner_of)
                                ):
                                    proposals.append(
                                        (
                                            1,
                                            1,
                                            _pair_score_sum([p]),
                                            int(diag),
                                            int(run[0][0]),
                                            int(run[-1][0]),
                                            [p],
                                        )
                                    )

        if not proposals:
            break

        proposals.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))

        accepted_any = False

        partner_now: Dict[int, int] = {}
        for x, y in pair_set:
            partner_now[x] = y
            partner_now[y] = x

        for _priority, _gap_len, _score_sum, _diag, _a0, _a1, add_pairs in proposals:
            conflict = False
            for a, b in add_pairs:
                if (a, b) in pair_set:
                    conflict = True
                    break
                if a in partner_now or b in partner_now:
                    conflict = True
                    break

            if conflict:
                continue

            if not _addition_preserves_existing_pairing(
                n=len(residues),
                existing_pairs=sorted(pair_set),
                add_pairs=add_pairs,
                allow_pseudoknot=cfg.allow_pseudoknot,
                require_same_level=require_same_level,
            ):
                continue

            for a, b in add_pairs:
                pair_set.add((a, b))
                partner_now[a] = b
                partner_now[b] = a

            accepted_any = True
            changed = True

            # Accept one repair, then rebuild runs from the new pair_set.
            break

        if not accepted_any:
            break

    return sorted(pair_set)

def rough_parse_ss_from_residues(
    residues: Sequence[ResidueCG],
    break_after: Sequence[bool],
    cfg: Optional[RoughSSConfig] = None,
) -> ParseResult:
    if cfg is None:
        cfg = RoughSSConfig()

    n = len(residues)
    base_candidates = _generate_scored_candidates(residues, cfg)

    forced_unpaired = set()

    final_kept_pairs: List[Tuple[int, int]] = []
    final_dbn_plain = "." * n
    final_pair_to_level: Dict[Tuple[int, int], int] = {}
    final_candidates: Dict[Tuple[int, int], PairCandidate] = base_candidates

    n_iter = max(1, int(getattr(cfg, "pk_cleanup_iters", 2)))
    if not getattr(cfg, "enable_pk_sandwich_cleanup", True):
        n_iter = 1

    for _it in range(n_iter):
        candidates = _filter_candidates_forced_unpaired(
            base_candidates,
            forced_unpaired=forced_unpaired,
        )
        final_candidates = candidates

        kept_pairs, dbn_plain, pair_to_level = _select_and_assign_once(
            n=n,
            residues=residues,
            candidates=candidates,
            cfg=cfg,
        )

        final_kept_pairs = kept_pairs
        final_dbn_plain = dbn_plain
        final_pair_to_level = pair_to_level

        newly_forced = _detect_pk_sandwich_forced_unpaired(
            kept_pairs=kept_pairs,
            pair_to_level=pair_to_level,
            cfg=cfg,
        )

        newly_forced = newly_forced - forced_unpaired
        if not newly_forced:
            break

        forced_unpaired.update(newly_forced)

    # ========================================================
    # Final post-processing only after normal parsing and all
    # pk-sandwich cleanup iterations are done.
    # ========================================================
    final_kept_pairs = _fill_single_gap_pairs_by_stem_context(
        residues=residues,
        kept_pairs=final_kept_pairs,
        candidates=final_candidates,
        cfg=cfg,
    )


    final_dbn_plain, final_pair_to_level = _assign_pairs_to_dbn(
        n=n,
        kept_pairs=final_kept_pairs,
        allow_pseudoknot=cfg.allow_pseudoknot,
    )

    final_kept_pairs = sorted([p for p in final_kept_pairs if p in final_pair_to_level])

    def insert_break_marks(dbn_str: str, b_after: Sequence[bool]) -> str:
        out: List[str] = []
        for idx, ch in enumerate(dbn_str):
            out.append(ch)
            if idx < len(dbn_str) - 1 and b_after[idx]:
                out.append("&")
        return "".join(out)

    dbn_with_breaks = insert_break_marks(final_dbn_plain, break_after)

    return ParseResult(
        dbn_plain=final_dbn_plain,
        dbn_with_breaks=dbn_with_breaks,
        kept_pairs=final_kept_pairs,
        break_after=list(break_after),
        residues=list(residues),
    )

def _make_unknown_residues(residues: Sequence[ResidueCG]) -> List[ResidueCG]:
    out: List[ResidueCG] = []
    for r in residues:
        rr = replace(r)
        rr.base = "N"
        rr.resname_raw = "N"
        rr.frame_R = None
        rr.frame_x = None
        rr.frame_y = None
        rr.frame_z = None
        out.append(rr)
    return out


def parse_cg_pdb_to_dbn_unknown_seq(
    pdb_path: str,
    cfg: Optional[RoughSSConfig] = None,
) -> ParseResult:

    if cfg is None:
        cfg = RoughSSConfig()

    residues = load_cg_residues_from_pdb(pdb_path)
    if not residues:
        raise ValueError("No recognizable CG residues found. Need at least one residue with P and/or C4' and/or N1/N9.")

    breaks = assign_breaks_and_fragments(residues, cfg)
    residues_unknown = _make_unknown_residues(residues)

    # Preserve fragment ids after break assignment.
    for rr, r0 in zip(residues_unknown, residues):
        rr.index = r0.index
        rr.fragment_id = r0.fragment_id

    return rough_parse_ss_from_residues(residues_unknown, breaks, cfg=cfg)


def parse_cg_pdb_to_dbn_native_seq(
    pdb_path: str,
    cfg: Optional[RoughSSConfig] = None,
) -> ParseResult:
    """Old native-sequence-aware behavior, kept for comparison/debug."""
    if cfg is None:
        cfg = RoughSSConfig(selection_mode="original")

    residues = load_cg_residues_from_pdb(pdb_path)
    if not residues:
        raise ValueError("No recognizable CG residues found. Need at least one residue with P and/or C4' and/or N1/N9.")

    breaks = assign_breaks_and_fragments(residues, cfg)
    return rough_parse_ss_from_residues(residues, breaks, cfg=cfg)


def parse_cg_pdb_to_dbn(
    pdb_path: str,
    cfg: Optional[RoughSSConfig] = None,
) -> ParseResult:
    """
    Public default parser.

    For DS3dRNA design-time auto-SS, default to the unknown-sequence parser so
    output does not depend on the native PDB sequence.
    """
    return parse_cg_pdb_to_dbn_unknown_seq(pdb_path, cfg=cfg)


# Backward-compatible aliases from previous experimental versions.
def parse_cg_pdb_to_dbn_noisy_seq_ensemble(
    pdb_path: str,
    cfg: Optional[RoughSSConfig] = None,
    noisy_cfg=None,
) -> ParseResult:
    # Noisy sequence has been deprecated in favor of deterministic unknown-seq parsing.
    return parse_cg_pdb_to_dbn_unknown_seq(pdb_path, cfg=cfg)


@dataclass
class NoisySeqSSConfig:
    # Kept only so older DS3dRNA.py imports do not fail.
    n_trials: int = 0
    seed: int = 12345
    min_pair_freq: float = 0.0
    selector: str = "deprecated_unknown_seq"
    verbose: bool = False


# ============================================================
# writer & sequence handling
# ============================================================

def sequence_with_breaks(residues: Sequence[ResidueCG], break_after: Sequence[bool]) -> str:
    seq: List[str] = []
    for r in residues:
        b = canonical_base(r.base)
        if b not in {"A", "U", "C", "G"}:
            b = "N"
        seq.append(b)

    out: List[str] = []
    for i, ch in enumerate(seq):
        out.append(ch)
        if i < len(seq) - 1 and break_after[i]:
            out.append("&")
    return "".join(out)


def write_dssr_like_dbn(out_path: str, title: str, seq_break: str, dbn_break: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f">{title}\n")
        f.write(seq_break + "\n")
        f.write(dbn_break + "\n")


# ============================================================
# CLI
# ============================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Native-sequence-independent CG SS parser with WC/nonWC windows, N1/N9 R/Y prior, stacking, and SO(3) refinement."
    )
    p.add_argument("pdb", help="Input CG PDB")
    p.add_argument("-o", "--out", default=None, help="Output .dbn path (default: <input>.dbn)")

    p.add_argument("--native-seq", action="store_true", help="Use old native-sequence-aware parser for comparison/debug.")
    p.add_argument("--selection-mode", choices=["maxpair", "original"], default="maxpair", help="Final selector. maxpair is recommended for design-time auto-SS.")

    p.add_argument("--no-gu", action="store_true", help="Disable GU wobble in native-seq mode")
    p.add_argument("--allow-noncanonical", action="store_true", help="Allow noncanonical base types in native-seq mode")
    p.add_argument("--min-seq-sep", type=int, default=4, help="Minimum same-chain separation")

    p.add_argument("--wc-nn-min", type=float, default=8.0, help="Minimum N-N distance for WC-like pair")
    p.add_argument("--wc-nn-max", type=float, default=9.5, help="Maximum N-N distance for WC-like pair")
    p.add_argument("--wc-nn-ideal", type=float, default=8.84, help="Ideal N-N distance for WC-like pair")

    p.add_argument("--nonwc-nn-min", type=float, default=7.5, help="Minimum N-N distance for non-WC-like pair")
    p.add_argument("--nonwc-nn-max", type=float, default=11.0, help="Maximum N-N distance for non-WC-like pair")

    p.add_argument("--min-score-gap", type=float, default=0.20, help="Minimum score gap over second-best partner in original mode")
    p.add_argument("--no-pseudoknot", action="store_true", help="Disable crossing pairs")
    p.add_argument("--break-c4-p", type=float, default=8.0, help="Break if dist(C4'(i), P(i+1)) > this")

    p.add_argument("--cross-tolerance", type=float, default=0.6, help="Extra distance tolerance for cross-chain")
    p.add_argument("--break-tolerance", type=float, default=0.3, help="Extra distance tolerance for chain breaks")
    p.add_argument("--c4-step", type=float, default=7.5, help="Max C4'-C4' distance to be considered continuous backbone")
    p.add_argument("--stacking-bonus", type=float, default=0.15, help="Score reduction for continuous base pairs")
    p.add_argument("--drop-isolated", action="store_true", help="Enable original pruning of weak isolated 1-bp stems")
    p.add_argument("--keep-isolated", action="store_true", help="Keep isolated pairs (default behavior)")
    p.add_argument("--keep-threshold", type=float, default=0.15, help="Score threshold to save good isolated pairs from pruning if --drop-isolated is used")

    p.add_argument("--disable-pk-sandwich-cleanup", action="store_true", help="Disable two-pass pseudoknot-sandwich forced-unpair cleanup")
    p.add_argument("--pk-cleanup-iters", type=int, default=2, help="Number of pseudoknot-sandwich cleanup iterations")
    p.add_argument("--pk-sandwich-min-pk", type=int, default=4, help="Minimum stacked PK pairs to trigger sandwich cleanup")
    p.add_argument("--pk-sandwich-max-inner", type=int, default=2, help="Maximum inner pairs removable by sandwich cleanup")
    p.add_argument("--pk-sandwich-min-ratio", type=float, default=2.0, help="Minimum pk/inner ratio to trigger sandwich cleanup")

    p.add_argument("--no-visibility-competition", action="store_true", help="Disable visibility-aware local competition")
    p.add_argument("--visibility-dot-min", type=float, default=-0.10, help="C4'->N visibility dot threshold for local competition")
    p.add_argument("--visibility-missing-invisible", action="store_true", help="Treat missing local frames as invisible in competition")

    p.add_argument("--no-so3", action="store_true", help="Disable SO(3)-based orientation refinement")
    p.add_argument("--so3-gamma", type=float, default=2.0, help="Exponent for trace-based rotational match")
    p.add_argument("--so3-beta", type=float, default=1.0, help="Exponent for face-to-face term")
    p.add_argument("--so3-lambda-local", type=float, default=0.90, help="Strength of local competition reweighting")
    p.add_argument("--so3-lambda-global", type=float, default=0.20, help="Strength of mild global orientation reweighting")
    p.add_argument("--so3-score-floor", type=float, default=0.05, help="Positive floor before multiplicative SO(3) reweight")
    p.add_argument("--so3-trigger-gap", type=float, default=0.30, help="Gap below this activates stronger local SO(3) competition")

    return p


def main():
    args = build_argparser().parse_args()

    cfg = RoughSSConfig(
        allow_gu_wobble=not args.no_gu,
        allow_noncanonical=args.allow_noncanonical,
        min_seq_sep=args.min_seq_sep,
        wc_nn_min=args.wc_nn_min,
        wc_nn_max=args.wc_nn_max,
        wc_nn_ideal=args.wc_nn_ideal,
        nonwc_nn_min=args.nonwc_nn_min,
        nonwc_nn_max=args.nonwc_nn_max,
        min_score_gap=args.min_score_gap,
        allow_pseudoknot=not args.no_pseudoknot,
        break_c4_to_p_max=args.break_c4_p,
        cross_chain_tolerance=args.cross_tolerance,
        chain_break_tolerance=args.break_tolerance,
        max_c4_step_dist=args.c4_step,
        stacking_bonus=args.stacking_bonus,
        keep_isolated_pairs=(False if args.drop_isolated else True),
        isolated_keep_threshold=args.keep_threshold,
        enable_pk_sandwich_cleanup=not args.disable_pk_sandwich_cleanup,
        pk_cleanup_iters=args.pk_cleanup_iters,
        pk_sandwich_min_pk_pairs=args.pk_sandwich_min_pk,
        pk_sandwich_max_inner_pairs=args.pk_sandwich_max_inner,
        pk_sandwich_min_ratio=args.pk_sandwich_min_ratio,
        use_visibility_competition=not args.no_visibility_competition,
        visibility_dot_min=args.visibility_dot_min,
        visibility_missing_is_visible=(False if args.visibility_missing_invisible else True),
        use_so3_refine=not args.no_so3,
        so3_gamma=args.so3_gamma,
        so3_beta=args.so3_beta,
        so3_lambda_local=args.so3_lambda_local,
        so3_lambda_global=args.so3_lambda_global,
        so3_score_floor=args.so3_score_floor,
        so3_trigger_gap=args.so3_trigger_gap,
        selection_mode=args.selection_mode,
    )

    if args.native_seq:
        result = parse_cg_pdb_to_dbn_native_seq(args.pdb, cfg=cfg)
        mode_label = "native-seq"
    else:
        result = parse_cg_pdb_to_dbn_unknown_seq(args.pdb, cfg=cfg)
        mode_label = "unknown-seq"

    out_path = args.out
    if out_path is None:
        root, _ = os.path.splitext(args.pdb)
        out_path = root + ".dbn"

    seq_break = sequence_with_breaks(result.residues, result.break_after)
    title = os.path.basename(args.pdb)

    write_dssr_like_dbn(
        out_path=out_path,
        title=title,
        seq_break=seq_break,
        dbn_break=result.dbn_with_breaks,
    )

    print(f"[OK] Wrote: {out_path}")
    print(f"[MODE] {mode_label}")
    print(f"[SELECTOR] {cfg.selection_mode}")
    print(f"[SEQ] {seq_break}")
    print(f"[DBN] {result.dbn_with_breaks}")
    print(f"[PAIR_COUNT] {len(result.kept_pairs)}")
    print(f"[SO3] {'ON' if cfg.use_so3_refine else 'OFF'}")


if __name__ == "__main__":
    main()
