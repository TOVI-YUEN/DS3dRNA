#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MidThermoPrior.py

Turner2004 local thermodynamic reranker for DS3dRNA mid-stage filtering.

Role in pipeline
----------------
Rough top32 -> MidThermoPrior top5 -> Fine

Design principles
-----------------
1. Local only. No multibranch/exterior/coaxial/global bookkeeping.
2. Use ViennaRNA-style Turner2004 `.par` file as the parameter backend.
3. Reuse existing DS SS interfaces when available:
   - inline dot-bracket
   - .dbn / text file containing dot-bracket
   - ``auto`` via ``Src.rough_ss_parser``
   - optional ``Src.ss_constraint`` helpers when importable
4. Conservative around `&`, crossing pairs, and complex topologies: skip them.
5. Compatible plugin-style API:
   - build_mid_thermo_prior(...)
   - build_seq_prior(...)
   - build_thermo_ss(...)
   - score_batch_idx(...)
   - select_topk_idx(...)

This module is NOT a full ViennaRNA evaluator. It is a local reranker that
uses the dominant Turner2004 local terms from a real `.par` file.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

# -----------------------------------------------------------------------------
# Optional imports from DS codebase
# -----------------------------------------------------------------------------
try:
    from Src.ss_constraint import (
        load_ss_string as _load_ss_string_external,
        effective_ss_length as _effective_ss_length_external,
        parse_ss_to_pairs as _parse_ss_to_pairs_external,
        looks_like_dotbracket as _looks_like_dotbracket_external,
    )
except Exception:  # pragma: no cover
    _load_ss_string_external = None
    _effective_ss_length_external = None
    _parse_ss_to_pairs_external = None
    _looks_like_dotbracket_external = None

try:
    from Src.coarse_grained_ss_parser import (
        parse_cg_pdb_to_dbn,
        RoughSSConfig,
        load_cg_residues_from_pdb,
    )
    HAS_ROUGH_SS = True
except Exception:  # pragma: no cover
    try:
        from Src.rough_ss_parser import parse_cg_pdb_to_dbn, RoughSSConfig, load_cg_residues_from_pdb
        HAS_ROUGH_SS = True
    except Exception:  # pragma: no cover
        parse_cg_pdb_to_dbn = None
        RoughSSConfig = None
        load_cg_residues_from_pdb = None
        HAS_ROUGH_SS = False


# -----------------------------------------------------------------------------
# Constants / helpers
# -----------------------------------------------------------------------------
R_KCAL = 0.00198720425864083  # kcal mol^-1 K^-1
T_REF_K = 310.15
DEFAULT_TOP_K = 5
DEFAULT_CACHE_LIMIT = 20000
INTERNAL_LOG_COEFF = 1.08

IDX_TO_BASE_RNA = {0: "A", 1: "U", 2: "C", 3: "G"}
BASES = {"A", "U", "C", "G"}
PURINES = {"A", "G"}
PYRIMIDINES = {"C", "U"}
CANONICAL_OR_GU = {"AU", "UA", "CG", "GC", "GU", "UG"}
AU_GU_ENDS = {"AU", "UA", "GU", "UG"}

AUTO_SS_PRESET = {
    "wc_nn_min": 8.0,
    "wc_nn_max": 9.5,
    "wc_nn_ideal": 8.84,
    "nonwc_nn_min": 7.5,
    "nonwc_nn_max": 10.0,
    "min_score_gap": 0.35,
    "allow_gu_wobble": True,
    "allow_pseudoknot": True,
}

# -----------------------------------------------------------------------------
# Dot-bracket / pseudoknot levels
# -----------------------------------------------------------------------------
# Supported levels:
#   1: ( )
#   2: [ ]
#   3: { }
#   4: < >
#   5+: A/a, B/b, ..., Z/z
#
# Note:
# - Uppercase letters are open brackets.
# - Lowercase letters are close brackets.
# - RNA bases in sequence are NOT parsed here; this is only for SS/DBN strings.
# - '&' is chain-break marker and is handled separately.
# -----------------------------------------------------------------------------

_LETTER_BRACKET_PAIRS = {chr(ord("A") + i): chr(ord("a") + i) for i in range(26)}

_BRACKET_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
    **_LETTER_BRACKET_PAIRS,
}

_CLOSE_TO_OPEN = {v: k for k, v in _BRACKET_PAIRS.items()}

_ALLOWED_DBN_CHARS = set(".:&") | set(_BRACKET_PAIRS.keys()) | set(_BRACKET_PAIRS.values())

# Escape symbols that are special inside regex char classes.
_DOTBRACKET_RE = re.compile(
    r"^["
    + re.escape("".join(sorted(_ALLOWED_DBN_CHARS)))
    + r"]+$"
)

_DEFAULT_PAR_REL = os.path.join("Turner2004_Par", "rna_turner2004.par")


# -----------------------------------------------------------------------------
# ViennaRNA Turner2004 `.par` file backend
# -----------------------------------------------------------------------------

class Turner04ParData:
    """
    Lightweight parser for ViennaRNA-style Turner2004 parameter files.

    It keeps the raw section content and also provides typed views for the
    local terms that MidThermoPrior actually consumes.
    """

    SECTION_KEYS = (
        "stack",
        "mismatch_hairpin",
        "mismatch_internal",
        "mismatch_internal_1n",
        "mismatch_internal_23",
        "mismatch_multi",
        "mismatch_exterior",
        "dangle5",
        "dangle3",
        "int11",
        "int21",
        "int22",
        "hairpin",
        "bulge",
        "interior",
        "ML_params",
        "NINIO",
        "Misc",
        "Hexaloops",
        "Tetraloops",
        "Triloops",
    )

    def __init__(self, filename: str):
        self.filename = os.path.abspath(os.path.expanduser(str(filename)))
        self.par_G = {k: {} for k in self.SECTION_KEYS}
        self.par_H = {k: {} for k in self.SECTION_KEYS if k not in {"ML_params", "NINIO", "Misc"}}
        self._load_from_file()

    @staticmethod
    def _strip_inline_comment(line: str) -> str:
        if "/*" in line:
            return line[:line.index("/*")].strip()
        return line.strip()

    @staticmethod
    def _centi_to_kcal(val: Any) -> float:
        sval = str(val).strip()
        if sval.upper() == "INF":
            return float("inf")
        return float(sval) / 100.0

    @staticmethod
    def _token_to_float_or_inf(val: Any) -> float:
        sval = str(val).strip()
        if sval.upper() == "INF":
            return float("inf")
        return float(sval)

    def _load_from_file(self) -> None:
        current_section = None
        current_lines: List[str] = []
        with open(self.filename, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.rstrip("\n")
                s = line.strip()
                if s.startswith("#"):
                    self._process_current_section(current_section, current_lines)
                    current_section = s
                    current_lines = []
                else:
                    current_lines.append(line)
            self._process_current_section(current_section, current_lines)

    def _target(self, param_type: str):
        return self.par_G if param_type == "G" else self.par_H

    def _process_current_section(self, section: Optional[str], lines: List[str]) -> None:
        if not section or not lines:
            return
        param_type = "H" if "_enthalpies" in section else "G"
        cleaned = section.replace("#", "").replace("_enthalpies", "").strip()
        if cleaned not in self.SECTION_KEYS:
            return

        if cleaned in {"stack", "dangle5", "dangle3"}:
            self._parse_matrix(lines, cleaned, param_type)
        elif cleaned in {"mismatch_hairpin", "mismatch_internal", "mismatch_internal_1n", "mismatch_internal_23", "mismatch_multi", "mismatch_exterior"}:
            self._parse_mismatch(lines, cleaned, param_type)
        elif cleaned == "int11":
            self._parse_int11(lines, param_type)
        elif cleaned == "int21":
            self._parse_int21(lines, param_type)
        elif cleaned == "int22":
            self._parse_int22(lines, param_type)
        elif cleaned in {"hairpin", "bulge", "interior", "ML_params", "NINIO", "Misc"}:
            self._parse_length_like(lines, cleaned, param_type)
        elif cleaned in {"Hexaloops", "Tetraloops", "Triloops"}:
            self._parse_special_loops(lines, cleaned)

    def _parse_matrix(self, lines: List[str], section_name: str, param_type: str) -> None:
        base_pairs = ("CG", "GC", "GU", "UG", "AU", "UA", "NN")
        bases = ("N", "A", "C", "G", "U")
        target = self._target(param_type)
        body = [ln for ln in lines if ln.strip()]
        if section_name == "stack":
            for j, line in enumerate(body[1:-1]):
                nums = self._strip_inline_comment(line).split()
                for i, value in enumerate(nums[:len(base_pairs)]):
                    target[section_name][f"{base_pairs[i]}_{base_pairs[j]}"] = value
        else:
            for j, line in enumerate(body[1:-1]):
                nums = self._strip_inline_comment(line).split()
                for i, value in enumerate(nums[:len(bases)]):
                    target[section_name][f"{bases[i]}_{base_pairs[j]}"] = value

    def _parse_mismatch(self, lines: List[str], section_name: str, param_type: str) -> None:
        use_E = section_name in {"mismatch_hairpin", "mismatch_internal", "mismatch_internal_1n"}
        cols = ("E", "A", "C", "G", "U") if use_E else ("N", "A", "C", "G", "U")
        target = self._target(param_type)
        for line in lines:
            s = line.strip()
            if not s or "/*" not in s:
                continue
            nums = self._strip_inline_comment(s).split()
            tag = s.split("/*", 1)[1].split("*/", 1)[0].strip()
            row = [p.strip() for p in tag.split(",")]
            if len(row) != 2:
                continue
            pair_type, left = row
            for i, value in enumerate(nums[:len(cols)]):
                right = cols[i]
                target[section_name][(pair_type, left, right)] = value

    def _parse_int11(self, lines: List[str], param_type: str) -> None:
        cols = ("N", "A", "C", "G", "U")
        target = self._target(param_type)
        sec = "int11"
        for line in lines:
            s = line.strip()
            if not s or "/*" not in s:
                continue
            nums = self._strip_inline_comment(s).split()
            tag = s.split("/*", 1)[1].split("*/", 1)[0].strip()
            row = [p.strip() for p in tag.split(",")]
            if len(row) != 3:
                continue
            outer, inner, left = row
            for i, value in enumerate(nums[:len(cols)]):
                right = cols[i]
                target[sec][(outer, inner, left, right)] = value

    def _parse_int21(self, lines: List[str], param_type: str) -> None:
        cols = ("N", "A", "C", "G", "U")
        target = self._target(param_type)
        sec = "int21"
        for line in lines:
            s = line.strip()
            if not s or "/*" not in s:
                continue
            nums = self._strip_inline_comment(s).split()
            tag = s.split("/*", 1)[1].split("*/", 1)[0].strip()
            row = [p.strip() for p in tag.split(",")]
            if len(row) != 4:
                continue
            outer, inner, a, b = row
            for i, value in enumerate(nums[:len(cols)]):
                c = cols[i]
                target[sec][(outer, inner, a, b, c)] = value

    def _parse_int22(self, lines: List[str], param_type: str) -> None:
        cols = ("A", "C", "G", "U")
        target = self._target(param_type)
        sec = "int22"
        for line in lines:
            s = line.strip()
            if not s or "/*" not in s:
                continue
            nums = self._strip_inline_comment(s).split()
            tag = s.split("/*", 1)[1].split("*/", 1)[0].strip()
            row = [p.strip() for p in tag.split(",")]
            if len(row) != 5:
                continue
            outer, inner, a, b, c = row
            for i, value in enumerate(nums[:len(cols)]):
                d = cols[i]
                target[sec][(outer, inner, a, b, c, d)] = value

    def _parse_length_like(self, lines: List[str], section_name: str, param_type: str) -> None:
        target = self._target(param_type)
        idx = 0
        for line in lines:
            s = self._strip_inline_comment(line)
            if not s:
                continue
            for tok in s.split():
                target[section_name][idx] = tok
                idx += 1

    def _parse_special_loops(self, lines: List[str], section_name: str) -> None:
        for line in lines:
            s = line.strip()
            if not s or s.startswith("/*"):
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            case, g, h = parts[:3]
            self.par_G[section_name][case] = g
            self.par_H[section_name][case] = h

    # -------- typed views used by MidThermoPrior --------
    def length_table(self, name: str) -> Dict[int, Tuple[float, Optional[float]]]:
        out: Dict[int, Tuple[float, Optional[float]]] = {}
        gtab = self.par_G.get(name, {})
        htab = self.par_H.get(name, {})
        for k, gv in gtab.items():
            dh = htab.get(k)
            out[int(k)] = (self._centi_to_kcal(gv), self._centi_to_kcal(dh) if dh is not None else None)
        return out

    def stack_table(self) -> Dict[Tuple[str, str], Tuple[float, float]]:
        out: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for k, gv in self.par_G["stack"].items():
            if "_" not in k:
                continue
            a, b = k.split("_", 1)
            if "N" in a or "N" in b:
                continue
            hv = self.par_H["stack"].get(k)
            if hv is None:
                continue
            out[(a, b)] = (self._centi_to_kcal(gv), self._centi_to_kcal(hv))
        return out

    def mismatch_table(self, name: str) -> Dict[Tuple[str, str, str], Tuple[float, float]]:
        out: Dict[Tuple[str, str, str], Tuple[float, float]] = {}
        gtab = self.par_G.get(name, {})
        htab = self.par_H.get(name, {})
        for key, gv in gtab.items():
            hv = htab.get(key)
            if hv is None:
                continue
            out[key] = (self._centi_to_kcal(gv), self._centi_to_kcal(hv))
        return out

    def int11_table(self) -> Dict[Tuple[str, str, str, str], Tuple[float, float]]:
        out: Dict[Tuple[str, str, str, str], Tuple[float, float]] = {}
        gtab = self.par_G.get("int11", {})
        htab = self.par_H.get("int11", {})
        for key, gv in gtab.items():
            hv = htab.get(key)
            if hv is None:
                continue
            out[key] = (self._centi_to_kcal(gv), self._centi_to_kcal(hv))
        return out

    def int21_table(self) -> Dict[Tuple[str, str, str, str, str], Tuple[float, float]]:
        out: Dict[Tuple[str, str, str, str, str], Tuple[float, float]] = {}
        gtab = self.par_G.get("int21", {})
        htab = self.par_H.get("int21", {})
        for key, gv in gtab.items():
            hv = htab.get(key)
            if hv is None:
                continue
            out[key] = (self._centi_to_kcal(gv), self._centi_to_kcal(hv))
        return out

    def int22_table(self) -> Dict[Tuple[str, str, str, str, str, str], Tuple[float, float]]:
        out: Dict[Tuple[str, str, str, str, str, str], Tuple[float, float]] = {}
        gtab = self.par_G.get("int22", {})
        htab = self.par_H.get("int22", {})
        for key, gv in gtab.items():
            hv = htab.get(key)
            if hv is None:
                continue
            out[key] = (self._centi_to_kcal(gv), self._centi_to_kcal(hv))
        return out

    def special_loops(self) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for sec in ("Triloops", "Tetraloops", "Hexaloops"):
            for seq, gv in self.par_G.get(sec, {}).items():
                hv = self.par_H.get(sec, {}).get(seq)
                if hv is None:
                    continue
                out[str(seq)] = (self._centi_to_kcal(gv), self._centi_to_kcal(hv))
        return out

    def terminal_au(self) -> Tuple[float, float]:
        misc = self.par_G.get("Misc", {})
        # Misc = DuplexInit(G,H), TerminalAU(G,H)
        dg = self._centi_to_kcal(misc.get(2, 50))
        dh = self._centi_to_kcal(misc.get(3, 370))
        return (dg, dh)

    def load_all_local(self) -> Dict[str, Any]:
        return {
            "stack": self.stack_table(),
            "mismatch_hairpin": self.mismatch_table("mismatch_hairpin"),
            "mismatch_internal": self.mismatch_table("mismatch_internal"),
            "mismatch_internal_1n": self.mismatch_table("mismatch_internal_1n"),
            "mismatch_internal_23": self.mismatch_table("mismatch_internal_23"),
            "int11": self.int11_table(),
            "int21": self.int21_table(),
            "int22": self.int22_table(),
            "hairpin": self.length_table("hairpin"),
            "bulge": self.length_table("bulge"),
            "interior": self.length_table("interior"),
            "special_loops": self.special_loops(),
            "terminal_au": self.terminal_au(),
        }


def _candidate_default_par_paths() -> List[str]:
    candidates: List[str] = []
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidates.append(os.path.join(here, _DEFAULT_PAR_REL))
    candidates.append(os.path.join(here, "rna_turner2004.par"))
    candidates.append(os.path.join(here, "Src", _DEFAULT_PAR_REL))
    return candidates


def _resolve_par_path(par_path: Optional[str]) -> str:
    if par_path:
        path = os.path.abspath(os.path.expanduser(str(par_path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Turner2004 par file not found: {path}")
        return path
    for cand in _candidate_default_par_paths():
        if os.path.isfile(cand):
            return cand
    # Return the canonical expected location for a helpful error message.
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), _DEFAULT_PAR_REL))


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _looks_like_dotbracket(s: str) -> bool:
    if _looks_like_dotbracket_external is not None:
        try:
            return bool(_looks_like_dotbracket_external(s))
        except Exception:
            pass
    return bool(str(s).strip()) and bool(_DOTBRACKET_RE.fullmatch(str(s).strip()))


def _extract_dotbracket_from_text(text: str) -> str:
    lines = [ln.rstrip("\n") for ln in str(text).splitlines()]
    records: List[Tuple[str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            j = i + 1
            payload: List[str] = []
            while j < len(lines) and len(payload) < 2:
                if lines[j].strip():
                    payload.append(lines[j].strip())
                j += 1
            if len(payload) == 2 and _looks_like_dotbracket(payload[1]):
                records.append((payload[0], payload[1]))
            i = j
        else:
            i += 1
    if records:
        return max(records, key=lambda x: len(x[1]))[1].strip()
    cand = [ln.strip() for ln in lines if _looks_like_dotbracket(ln.strip())]
    if not cand:
        raise ValueError("No dot-bracket-like content found")
    return max(cand, key=len)


def _normalize_dbn_with_breaks(dbn: str) -> Tuple[str, List[int], List[int]]:
    dbn = str(dbn).strip()
    compact: List[str] = []
    seg_ids: List[int] = []
    break_after: List[int] = []
    seg = 0
    residue_count = 0

    for ch in dbn:
        if ch == "&":
            if residue_count > 0:
                break_after.append(residue_count - 1)
            seg += 1
            continue

        if ch not in _ALLOWED_DBN_CHARS or ch == "&" or ch == ":":
            raise ValueError(
                f"Unsupported DBN character: {ch!r}. "
                "Supported symbols are . () [] {} <> A/a ... Z/z plus '&' chain breaks."
            )

        compact.append(ch)
        seg_ids.append(seg)
        residue_count += 1

    return "".join(compact), seg_ids, break_after


def _pair_table_from_dbn(compact_dbn: str) -> List[int]:
    """
    Parse compact DBN into pair table.

    Supports:
      Level 1: ( )
      Level 2: [ ]
      Level 3: { }
      Level 4: < >
      Level 5+: A/a, B/b, ..., Z/z

    Important:
      If letter pseudoknot levels are present, do NOT delegate to external
      parse_ss_to_pairs, because older DS ss_constraint parsers may only
      recognize the first four bracket families.
    """
    has_letter_levels = any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in compact_dbn)

    if (not has_letter_levels) and (_parse_ss_to_pairs_external is not None):
        try:
            pairs_ext = _parse_ss_to_pairs_external(compact_dbn)
            pair = [-1] * len(compact_dbn)
            for i, j in pairs_ext:
                pair[i] = j
                pair[j] = i
            return pair
        except Exception:
            pass

    stacks: Dict[str, List[int]] = {k: [] for k in _BRACKET_PAIRS}
    pair = [-1] * len(compact_dbn)

    for i, ch in enumerate(compact_dbn):
        if ch == ".":
            continue

        if ch in _BRACKET_PAIRS:
            stacks[ch].append(i)
            continue

        if ch in _CLOSE_TO_OPEN:
            op = _CLOSE_TO_OPEN[ch]
            if not stacks[op]:
                raise ValueError(f"Unbalanced DBN: closing {ch!r} at position {i}")
            j = stacks[op].pop()
            pair[i] = j
            pair[j] = i
            continue

        raise ValueError(
            f"Unsupported DBN symbol: {ch!r}. "
            "Supported symbols are . () [] {} <> A/a ... Z/z."
        )

    for op, st in stacks.items():
        if st:
            raise ValueError(f"Unbalanced DBN: dangling opener {op!r}")

    return pair


def _safe_interp_dg(dg37: float, dh: Optional[float], temp_k: float) -> float:
    if dh is None:
        return float(dg37)
    ds = (float(dh) - float(dg37)) / T_REF_K
    return float(dh) - temp_k * ds


def _same_segment(seg_ids: List[int], i: int, j: int) -> bool:
    return seg_ids[i] == seg_ids[j]


def _interval_same_segment(seg_ids: List[int], i: int, j: int) -> bool:
    if i > j:
        i, j = j, i
    sid = seg_ids[i]
    for k in range(i, j + 1):
        if seg_ids[k] != sid:
            return False
    return True


def _crossing(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    i, j = p1
    k, l = p2
    return (i < k < j < l) or (k < i < l < j)


def _seq_key_from_idx_row(row: Sequence[int]) -> str:
    return "".join(IDX_TO_BASE_RNA.get(int(x), "N") for x in row)


def _pair_type(a: str, b: str) -> str:
    return f"{a}{b}"


def _rev_pair(pair: str) -> str:
    return pair[::-1]


# -----------------------------------------------------------------------------
# Motif model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class _StackMotif:
    outer_i: int
    outer_j: int
    inner_i: int
    inner_j: int


@dataclass(frozen=True)
class _HairpinMotif:
    i: int
    j: int


@dataclass(frozen=True)
class _BulgeMotif:
    outer_i: int
    outer_j: int
    inner_i: int
    inner_j: int
    left_unpaired: int
    right_unpaired: int


@dataclass(frozen=True)
class _InternalMotif:
    outer_i: int
    outer_j: int
    inner_i: int
    inner_j: int
    left_unpaired: int
    right_unpaired: int


@dataclass
class _Context:
    dbn_raw: str
    dbn: str
    seg_ids: List[int]
    pair: List[int]
    pairs: List[Tuple[int, int]]
    crossing_pairs: set
    stacks: List[_StackMotif]
    hairpins: List[_HairpinMotif]
    bulges: List[_BulgeMotif]
    internals: List[_InternalMotif]
    skipped_pairs: int


# -----------------------------------------------------------------------------
# Main scorer
# -----------------------------------------------------------------------------

class MidThermoPrior:
    """
    Turner2004 local thermodynamic reranker driven by a ViennaRNA `.par` file.
    """

    def __init__(
        self,
        *,
        dbn: str,
        temperature_c: float = 37.0,
        top_k: int = DEFAULT_TOP_K,
        unsupported_pair_penalty: float = 0.0,
        cache_limit: int = DEFAULT_CACHE_LIMIT,
        verbose: bool = False,
        par_path: Optional[str] = None,
    ) -> None:
        self.temperature_c = float(temperature_c)
        self.temperature_k = 273.15 + self.temperature_c
        self.top_k = int(top_k)
        self.unsupported_pair_penalty = float(unsupported_pair_penalty)
        self.cache_limit = int(max(0, cache_limit))
        self.verbose = bool(verbose)
        self.par_path = _resolve_par_path(par_path)
        if not os.path.isfile(self.par_path):
            raise FileNotFoundError(
                f"Turner2004 par file not found. Expected at: {self.par_path}\n"
                "Either place rna_turner2004.par under Src/Turner2004_Par/ or pass par_path explicitly."
            )
        self.turner_par = Turner04ParData(self.par_path)
        self.tables = self.turner_par.load_all_local()

        for _name in ("hairpin", "bulge", "interior"):
            tab = self.tables.get(_name, {})
            finite_keys = []
            for k, v in tab.items():
                try:
                    if math.isfinite(float(v[0])):
                        finite_keys.append(int(k))
                except Exception:
                    pass
            if len(finite_keys) == 0:
                print(f"[WARN][thermo] Turner table '{_name}' has no finite entries; fallback path may be used.")

        compact_dbn, seg_ids, _ = _normalize_dbn_with_breaks(dbn)
        pair = _pair_table_from_dbn(compact_dbn)
        self.context = self._build_context(dbn, compact_dbn, seg_ids, pair)
        self.length = len(self.context.dbn)
        self._score_cache: Dict[str, Dict[str, float]] = {}

        # Fast tensor backend is opt-in and must remain semantically aligned with
        # the original string/reference path. The slow path remains the source
        # of truth, and only torch [B, N] batches are intercepted when enabled.
        self.enable_fast_tensor: bool = False
        self._dense_luts_by_device: Dict[str, Dict[str, Any]] = {}
        self._dense_lut_meta: Dict[str, Any] = {}
        self._fast_verify_samples: int = 0
        self._fast_verify_atol: float = 0.0
        self._fast_verify_rtol: float = 0.0
        self._fast_verify_warn_only: bool = True

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_ss_arg(
        cls,
        ss_arg: Optional[str],
        *,
        pdb_path: Optional[str] = None,
        temperature_c: float = 37.0,
        top_k: int = DEFAULT_TOP_K,
        unsupported_pair_penalty: float = 0.0,
        cache_limit: int = DEFAULT_CACHE_LIMIT,
        verbose: bool = False,
        par_path: Optional[str] = None,
    ) -> "MidThermoPrior":
        dbn = resolve_ss_arg_to_dbn(ss_arg=ss_arg, pdb_path=pdb_path)
        return cls(
            dbn=dbn,
            temperature_c=temperature_c,
            top_k=top_k,
            unsupported_pair_penalty=unsupported_pair_penalty,
            cache_limit=cache_limit,
            verbose=verbose,
            par_path=par_path,
        )

    # ------------------------------------------------------------------
    # interface compatibility aliases
    # ------------------------------------------------------------------
    def score_batch(self, cand_batch: Union[Sequence[str], Any], **kwargs):
        if torch is not None and isinstance(cand_batch, torch.Tensor):
            return self.score_batch_idx(cand_batch, **kwargs)
        if np is not None and isinstance(cand_batch, np.ndarray) and cand_batch.ndim == 2 and np.issubdtype(cand_batch.dtype, np.integer):
            return self.score_batch_idx(cand_batch, **kwargs)
        return self.score_batch_str(cand_batch, **kwargs)

    def select_topk(self, cand_batch: Union[Sequence[str], Any], top_k: Optional[int] = None, **kwargs):
        if torch is not None and isinstance(cand_batch, torch.Tensor):
            return self.select_topk_idx(cand_batch, top_k=top_k, **kwargs)
        if np is not None and isinstance(cand_batch, np.ndarray) and cand_batch.ndim == 2 and np.issubdtype(cand_batch.dtype, np.integer):
            return self.select_topk_idx(cand_batch, top_k=top_k, **kwargs)
        return self.select_topk_str(cand_batch, top_k=top_k, **kwargs)


    def set_fast_tensor(
        self,
        enabled: bool = True,
        *,
        verify_samples: int = 0,
        atol: float = 0.0,
        rtol: float = 0.0,
        warn_only: bool = True,
    ) -> None:
        """
        Enable or disable the dense-LUT tensor backend.

        The backend is designed to preserve the original Python/string semantics
        by pre-expanding all small Turner lookup tables into dense tensors using
        the original lookup helpers themselves as the semantic oracle.

        Parameters
        ----------
        enabled:
            Whether to use the fast backend for torch [B, N] integer batches.
        verify_samples:
            If >0, compare the fast result against the reference path on up to
            this many rows per call to control peak memory use.
        atol, rtol:
            Tolerances used by the optional verification. Set both to 0.0 for an
            exact check in float64 arithmetic.
        warn_only:
            If True, a mismatch causes a warning-style fallback behavior. If
            False, the mismatch is raised immediately.
        """
        self.enable_fast_tensor = bool(enabled)
        self._fast_verify_samples = int(max(0, verify_samples))
        self._fast_verify_atol = float(atol)
        self._fast_verify_rtol = float(rtol)
        self._fast_verify_warn_only = bool(warn_only)

    # ------------------------------------------------------------------
    # batch scoring
    # ------------------------------------------------------------------
    def score_one_seq_str(self, seq: str, return_breakdown: bool = False):
        seq = str(seq).strip().upper().replace("T", "U")
        if len(seq) != self.length:
            raise ValueError(f"Sequence length mismatch: got {len(seq)}, expected {self.length}")
        if any(ch not in BASES for ch in seq):
            raise ValueError(f"Sequence contains unsupported RNA symbols: {seq!r}")

        if seq in self._score_cache:
            bd = self._score_cache[seq]
            return dict(bd) if return_breakdown else bd["total_dG"]

        bd = self._score_sequence(seq)
        if self.cache_limit > 0:
            if len(self._score_cache) >= self.cache_limit:
                try:
                    self._score_cache.pop(next(iter(self._score_cache)))
                except Exception:
                    self._score_cache.clear()
            self._score_cache[seq] = bd
        return dict(bd) if return_breakdown else bd["total_dG"]

    def score_batch_str(self, seq_list: Sequence[str], return_breakdown: bool = False):
        if return_breakdown:
            return [self.score_one_seq_str(s, return_breakdown=True) for s in seq_list]
        scores = [float(self.score_one_seq_str(s, return_breakdown=False)) for s in seq_list]
        if np is not None:
            return np.asarray(scores, dtype=float)
        return scores

    def score_batch_idx(self, cand_idx_batch: Any, return_breakdown: bool = False):
        if torch is not None and isinstance(cand_idx_batch, torch.Tensor):
            if self.enable_fast_tensor and (not return_breakdown) and cand_idx_batch.ndim == 2:
                try:
                    scores = self._score_batch_idx_torch_fast(cand_idx_batch)
                    if self._fast_verify_samples > 0:
                        self._verify_fast_scores(cand_idx_batch, scores)
                    return scores
                except Exception as e:
                    msg = f"[WARN][thermo] fast tensor backend failed: {e}; falling back to reference path"
                    if self._fast_verify_warn_only:
                        print(msg)
                    else:
                        raise

            rows = cand_idx_batch.detach().cpu().tolist()
            seqs = [_seq_key_from_idx_row(r) for r in rows]
            if return_breakdown:
                return [self.score_one_seq_str(s, return_breakdown=True) for s in seqs]
            scores = [float(self.score_one_seq_str(s, return_breakdown=False)) for s in seqs]
            return torch.as_tensor(scores, dtype=torch.float32, device=cand_idx_batch.device)

        if np is not None and isinstance(cand_idx_batch, np.ndarray):
            rows = cand_idx_batch.tolist()
            seqs = [_seq_key_from_idx_row(r) for r in rows]
            if return_breakdown:
                return [self.score_one_seq_str(s, return_breakdown=True) for s in seqs]
            return np.asarray([float(self.score_one_seq_str(s)) for s in seqs], dtype=float)

        rows = list(cand_idx_batch)
        seqs = [_seq_key_from_idx_row(r) for r in rows]
        return self.score_batch_str(seqs, return_breakdown=return_breakdown)

    # ------------------------------------------------------------------
    # top-k selection
    # ------------------------------------------------------------------
    def select_topk_str(
        self,
        seq_list: Sequence[str],
        top_k: Optional[int] = None,
        return_scores: bool = True,
        return_breakdown: bool = False,
    ):
        k = int(self.top_k if top_k is None else top_k)
        if return_breakdown:
            breakdown = self.score_batch_str(seq_list, return_breakdown=True)
            scores = [float(x["total_dG"]) for x in breakdown]
        else:
            scores = list(map(float, self.score_batch_str(seq_list, return_breakdown=False)))
            breakdown = None
        order = sorted(range(len(scores)), key=lambda i: (scores[i], i))[:k]
        if return_scores and return_breakdown:
            return order, scores, breakdown
        if return_scores:
            return order, scores
        return order

    def select_topk_idx(
        self,
        cand_idx_batch: Any,
        top_k: Optional[int] = None,
        return_scores: bool = True,
        return_breakdown: bool = False,
    ):
        k = int(self.top_k if top_k is None else top_k)
        scores = self.score_batch_idx(cand_idx_batch, return_breakdown=False)

        if torch is not None and isinstance(scores, torch.Tensor):
            keep = torch.argsort(scores, stable=True)[:k]
            if return_breakdown:
                breakdown = self.score_batch_idx(cand_idx_batch, return_breakdown=True)
                if return_scores:
                    return keep, scores, breakdown
                return keep, breakdown
            if return_scores:
                return keep, scores
            return keep

        if np is not None and isinstance(scores, np.ndarray):
            keep = np.argsort(scores, kind="stable")[:k]
            if return_breakdown:
                breakdown = self.score_batch_idx(cand_idx_batch, return_breakdown=True)
                if return_scores:
                    return keep, scores, breakdown
                return keep, breakdown
            if return_scores:
                return keep, scores
            return keep

        scores_list = list(map(float, scores))
        keep = sorted(range(len(scores_list)), key=lambda i: (scores_list[i], i))[:k]
        if return_breakdown:
            breakdown = self.score_batch_idx(cand_idx_batch, return_breakdown=True)
            if return_scores:
                return keep, scores_list, breakdown
            return keep, breakdown
        if return_scores:
            return keep, scores_list
        return keep

    # ------------------------------------------------------------------
    # context building
    # ------------------------------------------------------------------
    def _build_context(self, raw_dbn: str, compact_dbn: str, seg_ids: List[int], pair: List[int]) -> _Context:
        pairs = [(i, j) for i, j in enumerate(pair) if j > i]
        crossing = set()
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                if _crossing(pairs[a], pairs[b]):
                    crossing.add(pairs[a])
                    crossing.add(pairs[b])

        direct_children: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        noncross_pairs = [p for p in pairs if p not in crossing]
        for p in noncross_pairs:
            i, j = p
            inside = [q for q in noncross_pairs if i < q[0] < q[1] < j]
            kids = []
            for q in inside:
                q_i, q_j = q
                contained = False
                for r in inside:
                    if r == q:
                        continue
                    if r[0] < q_i and q_j < r[1]:
                        contained = True
                        break
                if not contained:
                    kids.append(q)
            direct_children[p] = sorted(kids)

        stacks: List[_StackMotif] = []
        hairpins: List[_HairpinMotif] = []
        bulges: List[_BulgeMotif] = []
        internals: List[_InternalMotif] = []
        skipped = 0

        for p in noncross_pairs:
            i, j = p
            if not _same_segment(seg_ids, i, j):
                skipped += 1
                continue

            kids = direct_children.get(p, [])
            if len(kids) == 0:
                if _interval_same_segment(seg_ids, i, j):
                    hairpins.append(_HairpinMotif(i, j))
                else:
                    skipped += 1
                continue

            if len(kids) > 1:
                skipped += 1
                continue

            (k, l) = kids[0]
            if not (_same_segment(seg_ids, i, k) and _same_segment(seg_ids, l, j) and _same_segment(seg_ids, k, l)):
                skipped += 1
                continue

            a = k - i - 1
            b = j - l - 1
            if a < 0 or b < 0:
                skipped += 1
                continue

            if a == 0 and b == 0:
                stacks.append(_StackMotif(i, j, k, l))
            elif a == 0 or b == 0:
                bulges.append(_BulgeMotif(i, j, k, l, a, b))
            else:
                internals.append(_InternalMotif(i, j, k, l, a, b))

        return _Context(
            dbn_raw=raw_dbn,
            dbn=compact_dbn,
            seg_ids=seg_ids,
            pair=pair,
            pairs=pairs,
            crossing_pairs=crossing,
            stacks=stacks,
            hairpins=hairpins,
            bulges=bulges,
            internals=internals,
            skipped_pairs=skipped + len(crossing),
        )

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------
    def _score_sequence(self, seq: str) -> Dict[str, float]:
        bd = {
            "stack_dG": 0.0,
            "hairpin_dG": 0.0,
            "bulge_dG": 0.0,
            "internal_dG": 0.0,
            "unsupported_penalty": 0.0,
            "n_stack": 0.0,
            "n_hairpin": 0.0,
            "n_bulge": 0.0,
            "n_internal": 0.0,
            "n_skipped_complex": float(self.context.skipped_pairs),
        }

        for m in self.context.stacks:
            val, pen = self._score_stack(seq, m)
            bd["stack_dG"] += val
            bd["unsupported_penalty"] += pen
            bd["n_stack"] += 1.0

        for m in self.context.hairpins:
            val, pen = self._score_hairpin(seq, m)
            bd["hairpin_dG"] += val
            bd["unsupported_penalty"] += pen
            bd["n_hairpin"] += 1.0

        for m in self.context.bulges:
            val, pen = self._score_bulge(seq, m)
            bd["bulge_dG"] += val
            bd["unsupported_penalty"] += pen
            bd["n_bulge"] += 1.0

        for m in self.context.internals:
            val, pen = self._score_internal(seq, m)
            bd["internal_dG"] += val
            bd["unsupported_penalty"] += pen
            bd["n_internal"] += 1.0

        bd["total_dG"] = (
            bd["stack_dG"]
            + bd["hairpin_dG"]
            + bd["bulge_dG"]
            + bd["internal_dG"]
            + bd["unsupported_penalty"]
        )
        return bd

    # ------------------------ motif scorers ------------------------
    def _score_stack(self, seq: str, m: _StackMotif) -> Tuple[float, float]:
        top = seq[m.outer_i] + seq[m.inner_i]
        bot = seq[m.outer_j] + seq[m.inner_j]
        param = self.tables["stack"].get((top, bot))
        if param is None:
            return 0.0, self._unsupported_penalty(top[0] + bot[0]) + self._unsupported_penalty(top[1] + bot[1])
        return _safe_interp_dg(param[0], param[1], self.temperature_k), 0.0

    def _score_hairpin(self, seq: str, m: _HairpinMotif) -> Tuple[float, float]:
        i, j = m.i, m.j
        close = seq[i] + seq[j]
        n = j - i - 1
        if n < 3:
            return 1e6, 0.0

        full = seq[i : j + 1]
        if full in self.tables["special_loops"]:
            dg37, dh = self.tables["special_loops"][full]
            return _safe_interp_dg(dg37, dh, self.temperature_k), 0.0

        val = self._hairpin_initiation(n)
        pen = self._unsupported_penalty(close) if close not in CANONICAL_OR_GU else 0.0

        if n > 3:
            x = seq[i + 1]
            y = seq[j - 1]
            tmm = self._lookup_mismatch(self.tables["mismatch_hairpin"], close, x, y, allow_E=True)
            if tmm is not None:
                val += _safe_interp_dg(tmm[0], tmm[1], self.temperature_k)

        if close in AU_GU_ENDS:
            dg37, dh = self.tables["terminal_au"]
            val += _safe_interp_dg(dg37, dh, self.temperature_k)

        return val, pen


    def _score_bulge(self, seq: str, m: _BulgeMotif) -> Tuple[float, float]:
        n = m.left_unpaired + m.right_unpaired
        outer = seq[m.outer_i] + seq[m.outer_j]
        inner = seq[m.inner_i] + seq[m.inner_j]
        pen = 0.0
        if outer not in CANONICAL_OR_GU:
            pen += self._unsupported_penalty(outer)
        if inner not in CANONICAL_OR_GU:
            pen += self._unsupported_penalty(inner)
        if n <= 0:
            return 0.0, pen

        val = self._bulge_initiation(n)

        if n == 1:
            stk, stk_pen = self._score_stack(seq, _StackMotif(m.outer_i, m.outer_j, m.inner_i, m.inner_j))
            val += stk
            pen += stk_pen

            bulged_idx = None
            if m.left_unpaired == 1 and m.right_unpaired == 0:
                bulged_idx = m.outer_i + 1
            elif m.left_unpaired == 0 and m.right_unpaired == 1:
                bulged_idx = m.outer_j - 1

            if bulged_idx is not None:
                bulged_base = seq[bulged_idx]
                if bulged_base == "C":
                    left_nb = seq[bulged_idx - 1] if bulged_idx - 1 >= 0 else None
                    right_nb = seq[bulged_idx + 1] if bulged_idx + 1 < len(seq) else None
                    if left_nb == "C" or right_nb == "C":
                        val += self._special_c_bulge_bonus()

            val += self._bulge_migration_entropy(seq, m)

        else:
            val += self._bulge_au_gu_closure(outer)
            val += self._bulge_au_gu_closure(inner)

        return val, pen
    def _score_internal(self, seq: str, m: _InternalMotif) -> Tuple[float, float]:
        a = m.left_unpaired
        b = m.right_unpaired
        n = a + b
        outer = seq[m.outer_i] + seq[m.outer_j]
        inner = seq[m.inner_i] + seq[m.inner_j]

        pen = 0.0
        if outer not in CANONICAL_OR_GU:
            pen += self._unsupported_penalty(outer)
        if inner not in CANONICAL_OR_GU:
            pen += self._unsupported_penalty(inner)

        # 1x1 explicit table
        if a == 1 and b == 1:
            left = seq[m.outer_i + 1]
            right = seq[m.outer_j - 1]
            param = self._lookup_int11(outer, inner, left, right)
            if param is not None:
                return _safe_interp_dg(param[0], param[1], self.temperature_k), pen

        # 1x2 / 2x1 explicit table
        if (a, b) in {(1, 2), (2, 1)}:
            param = self._lookup_int21(seq, m, outer, inner)
            if param is not None:
                return _safe_interp_dg(param[0], param[1], self.temperature_k), pen

        # 2x2 explicit table
        if a == 2 and b == 2:
            param = self._lookup_int22(seq, m, outer, inner)
            if param is not None:
                return _safe_interp_dg(param[0], param[1], self.temperature_k), pen

        # Generic local approximation with finer mismatch families from par file.
        val = self._internal_initiation(n)
        val += self._internal_au_gu_closure(outer)
        val += self._internal_au_gu_closure(inner)
        val += self._internal_asymmetry(abs(a - b))

        x1 = seq[m.outer_i + 1]
        y1 = seq[m.outer_j - 1]
        x2 = seq[m.inner_i - 1]
        y2 = seq[m.inner_j + 1]

        if min(a, b) == 1 and max(a, b) >= 2:
            tab = self.tables["mismatch_internal_1n"]
        elif {a, b} == {2, 3}:
            tab = self.tables["mismatch_internal_23"]
        else:
            tab = self.tables["mismatch_internal"]

        mm1 = self._lookup_mismatch(tab, outer, x1, y1, allow_E=(tab is not self.tables["mismatch_internal_23"]))
        if mm1 is not None:
            val += _safe_interp_dg(mm1[0], mm1[1], self.temperature_k)
        mm2 = self._lookup_mismatch(tab, inner, y2, x2, allow_E=(tab is not self.tables["mismatch_internal_23"]))
        if mm2 is not None:
            val += _safe_interp_dg(mm2[0], mm2[1], self.temperature_k)

        return val, pen

    # ------------------------ lookup helpers ------------------------

    def _finite_anchor_key(self, table: Dict[int, Tuple[float, Optional[float]]], min_key: int) -> Optional[int]:
        keys = []
        for k, v in table.items():
            try:
                kk = int(k)
                dg37 = float(v[0])
            except Exception:
                continue
            if kk >= min_key and math.isfinite(dg37):
                keys.append(kk)
        return max(keys) if keys else None

    def _unsupported_penalty(self, pair_type: str) -> float:
        if pair_type in CANONICAL_OR_GU:
            return 0.0
        return float(self.unsupported_pair_penalty)

    def _hairpin_initiation(self, n: int) -> float:
        table = self.tables["hairpin"]
        if n in table:
            dg37, dh = table[n]
            if math.isfinite(float(dg37)):
                return _safe_interp_dg(dg37, dh, self.temperature_k)

        anchor = self._finite_anchor_key(table, min_key=3)
        if anchor is None:
            if self.verbose:
                print("[WARN][thermo] hairpin table empty/non-finite; using 0.0 fallback")
            return 0.0

        dg37, dh = table[anchor]
        base = _safe_interp_dg(dg37, dh, self.temperature_k)
        return base + 1.75 * R_KCAL * self.temperature_k * math.log(float(n) / float(anchor))

    def _bulge_initiation(self, n: int) -> float:
        table = self.tables["bulge"]
        if n in table:
            dg37, dh = table[n]
            if math.isfinite(float(dg37)):
                return _safe_interp_dg(dg37, dh, self.temperature_k)

        anchor = self._finite_anchor_key(table, min_key=1)
        if anchor is None:
            if self.verbose:
                print("[WARN][thermo] bulge table empty/non-finite; using 0.0 fallback")
            return 0.0

        dg37, dh = table[anchor]
        base = _safe_interp_dg(dg37, dh, self.temperature_k)
        return base + 1.75 * R_KCAL * self.temperature_k * math.log(float(n) / float(anchor))

    def _internal_initiation(self, n: int) -> float:
        table = self.tables["interior"]
        if n in table:
            dg37, dh = table[n]
            if math.isfinite(float(dg37)):
                return _safe_interp_dg(dg37, dh, self.temperature_k)

        anchor = self._finite_anchor_key(table, min_key=2)
        if anchor is not None:
            dg37, dh = table[anchor]
            base = _safe_interp_dg(dg37, dh, self.temperature_k)
            return base + INTERNAL_LOG_COEFF * math.log(float(n) / float(anchor))

        fallback_37 = {2: 0.5, 3: 1.6, 4: 1.1, 5: 2.1, 6: 1.9}
        if n <= 1:
            return 0.0
        if n in fallback_37:
            if self.verbose:
                print(f"[WARN][thermo] interior table empty/non-finite; using Turner2004 fallback for n={n}")
            return float(fallback_37[n])

        if self.verbose:
            print(f"[WARN][thermo] interior table empty/non-finite; using Turner2004 log fallback for n={n}")
        return 1.9 + INTERNAL_LOG_COEFF * math.log(float(n) / 6.0)

    def _ninio_params(self) -> Tuple[float, float]:
        gtab = self.turner_par.par_G.get("NINIO", {})
        vals = []
        for k in sorted(gtab.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            try:
                vals.append(float(self.turner_par._centi_to_kcal(gtab[k])))
            except Exception:
                continue
        slope = vals[0] if len(vals) >= 1 and math.isfinite(vals[0]) else 0.6
        cap = vals[1] if len(vals) >= 2 and math.isfinite(vals[1]) else 3.0
        return float(slope), float(cap)

    def _internal_asymmetry(self, delta: int) -> float:
        if delta <= 0:
            return 0.0
        slope, cap = self._ninio_params()
        return min(float(cap), float(slope) * float(delta))

    def _internal_au_gu_closure(self, close_pair: str) -> float:
        return 0.7 if close_pair in AU_GU_ENDS else 0.0

    def _bulge_au_gu_closure(self, close_pair: str) -> float:
        return 0.7 if close_pair in AU_GU_ENDS else 0.0

    def _special_c_bulge_bonus(self) -> float:
        return -0.9

    def _bulge_migration_entropy(self, seq: str, m: _BulgeMotif) -> float:
        n = m.left_unpaired + m.right_unpaired
        if n != 1:
            return 0.0

        if m.left_unpaired == 1 and m.right_unpaired == 0:
            bulged_idx = m.outer_i + 1
            step = 1
        elif m.left_unpaired == 0 and m.right_unpaired == 1:
            bulged_idx = m.outer_j - 1
            step = -1
        else:
            return 0.0

        if bulged_idx < 0 or bulged_idx >= len(seq):
            return 0.0

        base = seq[bulged_idx]
        states = 1
        k = bulged_idx + step
        while 0 <= k < len(seq) and seq[k] == base:
            states += 1
            k += step

        if states <= 1:
            return 0.0
        return -R_KCAL * self.temperature_k * math.log(float(states))
    def _lookup_mismatch(
        self,
        table: Dict[Tuple[str, str, str], Tuple[float, float]],
        close: str,
        left: str,
        right: str,
        allow_E: bool,
    ) -> Optional[Tuple[float, float]]:
        candidates: List[Tuple[str, str, str]] = [
            (close, left, right),
            (_rev_pair(close), right, left),
        ]
        if allow_E:
            candidates.extend([
                (close, "E", right),
                (close, left, "E"),
                (_rev_pair(close), "E", left),
                (_rev_pair(close), right, "E"),
            ])
        else:
            candidates.extend([
                (close, "N", right),
                (close, left, "N"),
                (_rev_pair(close), "N", left),
                (_rev_pair(close), right, "N"),
            ])
        for key in candidates:
            if key in table:
                return table[key]
        return None

    def _lookup_int11(self, outer: str, inner: str, left: str, right: str) -> Optional[Tuple[float, float]]:
        table = self.tables["int11"]
        candidates = [
            (outer, inner, left, right),
            (_rev_pair(inner), _rev_pair(outer), right, left),
            (inner, outer, right, left),
            (_rev_pair(outer), _rev_pair(inner), left, right),
            (outer, inner, "N", right),
            (outer, inner, left, "N"),
            (_rev_pair(inner), _rev_pair(outer), "N", left),
            (_rev_pair(inner), _rev_pair(outer), right, "N"),
        ]
        for key in candidates:
            if key in table:
                return table[key]
        return None

    def _lookup_int21(self, seq: str, m: _InternalMotif, outer: str, inner: str) -> Optional[Tuple[float, float]]:
        table = self.tables["int21"]
        left_bases = [seq[m.outer_i + t] for t in range(1, m.left_unpaired + 1)]
        right_bases = [seq[m.outer_j - t] for t in range(1, m.right_unpaired + 1)]
        candidates: List[Tuple[str, str, str, str, str]] = []
        if m.left_unpaired == 1 and m.right_unpaired == 2:
            l1 = left_bases[0]
            r1, r2 = right_bases
            candidates.extend([
                (outer, inner, l1, r1, r2),
                (outer, inner, l1, r2, r1),
                (outer, inner, r1, r2, l1),
                (outer, inner, r2, r1, l1),
            ])
        elif m.left_unpaired == 2 and m.right_unpaired == 1:
            l1, l2 = left_bases
            r1 = right_bases[0]
            candidates.extend([
                (outer, inner, l1, l2, r1),
                (outer, inner, l2, l1, r1),
                (outer, inner, r1, l1, l2),
                (outer, inner, r1, l2, l1),
            ])
        # mirrored / reversed variants
        extra: List[Tuple[str, str, str, str, str]] = []
        for op, ip, a, b, c in candidates:
            extra.extend([
                (_rev_pair(ip), _rev_pair(op), c, b, a),
                (ip, op, c, b, a),
                (_rev_pair(op), _rev_pair(ip), a, b, c),
            ])
        candidates.extend(extra)
        # N-backed fallbacks
        fb: List[Tuple[str, str, str, str, str]] = []
        for op, ip, a, b, c in candidates:
            fb.extend([
                (op, ip, "N", b, c),
                (op, ip, a, "N", c),
                (op, ip, a, b, "N"),
            ])
        candidates.extend(fb)
        for key in candidates:
            if key in table:
                return table[key]
        return None

    def _lookup_int22(self, seq: str, m: _InternalMotif, outer: str, inner: str) -> Optional[Tuple[float, float]]:
        table = self.tables["int22"]
        l1, l2 = [seq[m.outer_i + t] for t in range(1, 3)]
        r1, r2 = [seq[m.outer_j - t] for t in range(1, 3)]
        candidates: List[Tuple[str, str, str, str, str, str]] = [
            (outer, inner, l1, l2, r2, r1),
            (outer, inner, l2, l1, r2, r1),
            (outer, inner, l1, l2, r1, r2),
            (outer, inner, r2, r1, l2, l1),
            (outer, inner, r1, r2, l2, l1),
        ]
        extra: List[Tuple[str, str, str, str, str, str]] = []
        for op, ip, a, b, c, d in candidates:
            extra.extend([
                (_rev_pair(ip), _rev_pair(op), d, c, b, a),
                (ip, op, d, c, b, a),
                (_rev_pair(op), _rev_pair(ip), a, b, c, d),
            ])
        candidates.extend(extra)
        for key in candidates:
            if key in table:
                return table[key]
        return None

    # ------------------------------------------------------------------
    # dense-LUT fast tensor backend
    # ------------------------------------------------------------------
    def _verify_fast_scores(self, cand_idx_batch: "torch.Tensor", fast_scores: "torch.Tensor") -> None:
        if torch is None:
            return
        B = int(cand_idx_batch.shape[0])
        if B <= 0:
            return
        k = min(B, int(self._fast_verify_samples))
        if k <= 0:
            return
        sample = cand_idx_batch[:k]
        rows = sample.detach().cpu().tolist()
        seqs = [_seq_key_from_idx_row(r) for r in rows]
        ref = torch.as_tensor(
            [float(self.score_one_seq_str(s, return_breakdown=False)) for s in seqs],
            dtype=torch.float64,
            device=fast_scores.device,
        )
        got = fast_scores[:k].to(dtype=torch.float64)
        if not torch.allclose(got, ref, atol=self._fast_verify_atol, rtol=self._fast_verify_rtol):
            diff = torch.max(torch.abs(got - ref)).item()
            msg = (
                f"fast tensor verification mismatch: max_abs_diff={diff:.6g}, "
                f"atol={self._fast_verify_atol}, rtol={self._fast_verify_rtol}"
            )
            raise RuntimeError(msg)

    def _get_dense_luts_for_device(self, device: "torch.device") -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for fast tensor backend")
        key = str(device)
        if key in self._dense_luts_by_device:
            return self._dense_luts_by_device[key]

        dtype = torch.float64
        idx_to_pair = {0: "CG", 1: "GC", 2: "GU", 3: "UG", 4: "AU", 5: "UA", 6: "NN"}
        idx_to_base = {0: "A", 1: "U", 2: "C", 3: "G"}
        base_to_idx = {"A": 0, "U": 1, "C": 2, "G": 3}

        def interp_param(param: Optional[Tuple[float, float]]) -> float:
            if param is None:
                return float("nan")
            return float(_safe_interp_dg(param[0], param[1], self.temperature_k))

        luts: Dict[str, Any] = {}

        pair_code = torch.full((4, 4), 6, dtype=torch.long, device=device)
        pair_pen = torch.full((4, 4), float(self.unsupported_pair_penalty), dtype=dtype, device=device)
        mapping = {"CG": 0, "GC": 1, "GU": 2, "UG": 3, "AU": 4, "UA": 5}
        for pair_str, code in mapping.items():
            ai, bi = base_to_idx[pair_str[0]], base_to_idx[pair_str[1]]
            pair_code[ai, bi] = code
            pair_pen[ai, bi] = 0.0
        luts["pair_code"] = pair_code
        luts["pair_pen"] = pair_pen

        stack = torch.full((7, 7), float("nan"), dtype=dtype, device=device)
        for p1 in range(7):
            for p2 in range(7):
                stack[p1, p2] = interp_param(self.tables["stack"].get((idx_to_pair[p1], idx_to_pair[p2])))
        luts["stack"] = stack

        def build_mismatch(tab_name: str, allow_E: bool) -> torch.Tensor:
            lut = torch.full((7, 4, 4), float("nan"), dtype=dtype, device=device)
            table = self.tables[tab_name]
            for p in range(7):
                close = idx_to_pair[p]
                for l in range(4):
                    for r in range(4):
                        param = self._lookup_mismatch(table, close, idx_to_base[l], idx_to_base[r], allow_E=allow_E)
                        lut[p, l, r] = interp_param(param)
            return lut

        luts["mm_hairpin"] = build_mismatch("mismatch_hairpin", True)
        luts["mm_internal"] = build_mismatch("mismatch_internal", True)
        luts["mm_internal_1n"] = build_mismatch("mismatch_internal_1n", True)
        luts["mm_internal_23"] = build_mismatch("mismatch_internal_23", False)

        int11 = torch.full((7, 7, 4, 4), float("nan"), dtype=dtype, device=device)
        for op in range(7):
            outer = idx_to_pair[op]
            for ip in range(7):
                inner = idx_to_pair[ip]
                for l in range(4):
                    for r in range(4):
                        param = self._lookup_int11(outer, inner, idx_to_base[l], idx_to_base[r])
                        int11[op, ip, l, r] = interp_param(param)
        luts["int11"] = int11

        int21_1x2 = torch.full((7, 7, 4, 4, 4), float("nan"), dtype=dtype, device=device)
        int21_2x1 = torch.full((7, 7, 4, 4, 4), float("nan"), dtype=dtype, device=device)
        dummy_1x2 = _InternalMotif(0, 9, 2, 6, 1, 2)
        dummy_2x1 = _InternalMotif(0, 9, 3, 7, 2, 1)
        for op in range(7):
            outer = idx_to_pair[op]
            for ip in range(7):
                inner = idx_to_pair[ip]
                for b1 in range(4):
                    for b2 in range(4):
                        for b3 in range(4):
                            seq_1x2 = ["A"] * 10
                            seq_1x2[1] = idx_to_base[b1]
                            seq_1x2[8] = idx_to_base[b2]
                            seq_1x2[7] = idx_to_base[b3]
                            p1 = self._lookup_int21("".join(seq_1x2), dummy_1x2, outer, inner)
                            int21_1x2[op, ip, b1, b2, b3] = interp_param(p1)

                            seq_2x1 = ["A"] * 10
                            seq_2x1[1] = idx_to_base[b1]
                            seq_2x1[2] = idx_to_base[b2]
                            seq_2x1[8] = idx_to_base[b3]
                            p2 = self._lookup_int21("".join(seq_2x1), dummy_2x1, outer, inner)
                            int21_2x1[op, ip, b1, b2, b3] = interp_param(p2)
        luts["int21_1x2"] = int21_1x2
        luts["int21_2x1"] = int21_2x1

        int22 = torch.full((7, 7, 4, 4, 4, 4), float("nan"), dtype=dtype, device=device)
        dummy_22 = _InternalMotif(0, 9, 3, 6, 2, 2)
        for op in range(7):
            outer = idx_to_pair[op]
            for ip in range(7):
                inner = idx_to_pair[ip]
                for l1 in range(4):
                    for l2 in range(4):
                        for r1 in range(4):
                            for r2 in range(4):
                                seq_22 = ["A"] * 10
                                seq_22[1], seq_22[2] = idx_to_base[l1], idx_to_base[l2]
                                seq_22[8], seq_22[7] = idx_to_base[r1], idx_to_base[r2]
                                param = self._lookup_int22("".join(seq_22), dummy_22, outer, inner)
                                int22[op, ip, l1, l2, r1, r2] = interp_param(param)
        luts["int22"] = int22

        special: Dict[int, Dict[str, torch.Tensor]] = {}
        for seq_str, (dg37, dh) in self.tables["special_loops"].items():
            seq_s = str(seq_str).strip().upper().replace("T", "U")
            if any(ch not in base_to_idx for ch in seq_s):
                continue
            L = len(seq_s)
            payload = special.setdefault(L, {"seqs": [], "vals": []})
            payload["seqs"].append([base_to_idx[ch] for ch in seq_s])
            payload["vals"].append(float(_safe_interp_dg(dg37, dh, self.temperature_k)))
        luts["special_loops"] = {
            L: {
                "seqs": torch.tensor(payload["seqs"], dtype=torch.long, device=device),
                "vals": torch.tensor(payload["vals"], dtype=dtype, device=device),
            }
            for L, payload in special.items()
        }

        luts["terminal_au"] = torch.tensor(float(_safe_interp_dg(*self.tables["terminal_au"], self.temperature_k)), dtype=dtype, device=device)
        luts["bulge_closure"] = torch.tensor(float(self._bulge_au_gu_closure("AU")), dtype=dtype, device=device)
        luts["internal_closure"] = torch.tensor(float(self._internal_au_gu_closure("AU")), dtype=dtype, device=device)
        luts["special_c_bulge_bonus"] = torch.tensor(float(self._special_c_bulge_bonus()), dtype=dtype, device=device)
        luts["rkT"] = torch.tensor(float(R_KCAL * self.temperature_k), dtype=dtype, device=device)

        self._dense_luts_by_device[key] = luts
        self._dense_lut_meta[key] = {"dtype": "float64"}
        return luts

    def _score_batch_idx_torch_fast(self, x: "torch.Tensor") -> "torch.Tensor":
        if torch is None:
            raise RuntimeError("torch is required for fast tensor backend")
        if x.ndim != 2:
            raise ValueError("cand_idx_batch must have shape [B, N]")

        x = x.long()
        B = int(x.shape[0])
        device = x.device
        luts = self._get_dense_luts_for_device(device)
        pair_code = luts["pair_code"]
        pair_pen = luts["pair_pen"]
        dtype = torch.float64

        scores = torch.zeros(B, dtype=dtype, device=device)
        penalties = torch.zeros(B, dtype=dtype, device=device)

        def get_pc(i: int, j: int) -> torch.Tensor:
            return pair_code[x[:, i], x[:, j]]

        def get_pen(i: int, j: int) -> torch.Tensor:
            return pair_pen[x[:, i], x[:, j]]

        canonical_mask_codes = lambda pc: (pc >= 2) & (pc <= 5)

        for m in self.context.stacks:
            pc1 = get_pc(m.outer_i, m.inner_i)
            pc2 = get_pc(m.outer_j, m.inner_j)
            e = luts["stack"][pc1, pc2]
            scores += torch.nan_to_num(e, nan=0.0)
            penalties += torch.where(torch.isnan(e), get_pen(m.outer_i, m.inner_i) + get_pen(m.outer_j, m.inner_j), torch.zeros_like(penalties))

        term_au = luts["terminal_au"]
        for m in self.context.hairpins:
            i, j = m.i, m.j
            n = j - i - 1
            if n < 3:
                scores += 1e6
                continue

            pc = get_pc(i, j)
            pen = get_pen(i, j)
            val = torch.full((B,), float(self._hairpin_initiation(n)), dtype=dtype, device=device)

            L = j - i + 1
            special_val: Optional[torch.Tensor] = None
            if L in luts["special_loops"]:
                sp = luts["special_loops"][L]
                seg = x[:, i:j+1]
                eq = (seg.unsqueeze(1) == sp["seqs"].unsqueeze(0)).all(dim=-1)
                has = eq.any(dim=1)
                idx = eq.long().argmax(dim=1)
                special_val = torch.where(has, sp["vals"][idx], torch.full((B,), float("nan"), dtype=dtype, device=device))

            if n > 3:
                val += torch.nan_to_num(luts["mm_hairpin"][pc, x[:, i + 1], x[:, j - 1]], nan=0.0)
            val += torch.where(canonical_mask_codes(pc), term_au, torch.zeros(B, dtype=dtype, device=device))

            if special_val is not None:
                use_special = ~torch.isnan(special_val)
                scores += torch.where(use_special, special_val, val)
                penalties += torch.where(use_special, torch.zeros_like(pen), pen)
            else:
                scores += val
                penalties += pen

        for m in self.context.bulges:
            n = m.left_unpaired + m.right_unpaired
            pc_out = get_pc(m.outer_i, m.outer_j)
            pc_in = get_pc(m.inner_i, m.inner_j)
            pen = get_pen(m.outer_i, m.outer_j) + get_pen(m.inner_i, m.inner_j)
            if n <= 0:
                penalties += pen
                continue

            val = torch.full((B,), float(self._bulge_initiation(n)), dtype=dtype, device=device)
            if n == 1:
                e = luts["stack"][get_pc(m.outer_i, m.inner_i), get_pc(m.outer_j, m.inner_j)]
                val += torch.nan_to_num(e, nan=0.0)
                pen = pen + torch.where(torch.isnan(e), get_pen(m.outer_i, m.inner_i) + get_pen(m.outer_j, m.inner_j), torch.zeros_like(pen))

                b_idx = m.outer_i + 1 if m.left_unpaired == 1 else m.outer_j - 1
                step = 1 if m.left_unpaired == 1 else -1

                base_c = (x[:, b_idx] == 2)
                nb_c = torch.zeros(B, dtype=torch.bool, device=device)
                if b_idx - 1 >= 0:
                    nb_c |= (x[:, b_idx - 1] == 2)
                if b_idx + 1 < self.length:
                    nb_c |= (x[:, b_idx + 1] == 2)
                val += torch.where(base_c & nb_c, luts["special_c_bulge_bonus"], torch.zeros(B, dtype=dtype, device=device))

                same = torch.ones(B, dtype=torch.long, device=device)
                active = torch.ones(B, dtype=torch.bool, device=device)
                k = b_idx + step
                while 0 <= k < self.length:
                    active &= (x[:, k] == x[:, b_idx])
                    same += active.long()
                    k += step
                same_f = same.to(dtype=dtype)
                mig_ent = -luts["rkT"] * torch.log(same_f)
                val += torch.where(same > 1, mig_ent, torch.zeros(B, dtype=dtype, device=device))
            else:
                val += torch.where(canonical_mask_codes(pc_out), luts["bulge_closure"], torch.zeros(B, dtype=dtype, device=device))
                val += torch.where(canonical_mask_codes(pc_in), luts["bulge_closure"], torch.zeros(B, dtype=dtype, device=device))

            scores += val
            penalties += pen

        for m in self.context.internals:
            a_n, b_n = m.left_unpaired, m.right_unpaired
            n = a_n + b_n
            pc_out = get_pc(m.outer_i, m.outer_j)
            pc_in = get_pc(m.inner_i, m.inner_j)
            pen = get_pen(m.outer_i, m.outer_j) + get_pen(m.inner_i, m.inner_j)

            explicit = torch.full((B,), float("nan"), dtype=dtype, device=device)
            if a_n == 1 and b_n == 1:
                explicit = luts["int11"][pc_out, pc_in, x[:, m.outer_i + 1], x[:, m.outer_j - 1]]
            elif a_n == 1 and b_n == 2:
                explicit = luts["int21_1x2"][pc_out, pc_in, x[:, m.outer_i + 1], x[:, m.outer_j - 1], x[:, m.outer_j - 2]]
            elif a_n == 2 and b_n == 1:
                explicit = luts["int21_2x1"][pc_out, pc_in, x[:, m.outer_i + 1], x[:, m.outer_i + 2], x[:, m.outer_j - 1]]
            elif a_n == 2 and b_n == 2:
                explicit = luts["int22"][pc_out, pc_in, x[:, m.outer_i + 1], x[:, m.outer_i + 2], x[:, m.outer_j - 1], x[:, m.outer_j - 2]]

            gen_val = torch.full((B,), float(self._internal_initiation(n) + self._internal_asymmetry(abs(a_n - b_n))), dtype=dtype, device=device)
            gen_val += torch.where(canonical_mask_codes(pc_out), luts["internal_closure"], torch.zeros(B, dtype=dtype, device=device))
            gen_val += torch.where(canonical_mask_codes(pc_in), luts["internal_closure"], torch.zeros(B, dtype=dtype, device=device))

            if min(a_n, b_n) == 1 and max(a_n, b_n) >= 2:
                lut_mm = luts["mm_internal_1n"]
            elif {a_n, b_n} == {2, 3}:
                lut_mm = luts["mm_internal_23"]
            else:
                lut_mm = luts["mm_internal"]

            gen_val += torch.nan_to_num(lut_mm[pc_out, x[:, m.outer_i + 1], x[:, m.outer_j - 1]], nan=0.0)
            gen_val += torch.nan_to_num(lut_mm[pc_in, x[:, m.inner_j + 1], x[:, m.inner_i - 1]], nan=0.0)

            scores += torch.where(torch.isnan(explicit), gen_val, explicit)
            penalties += pen

        out = scores + penalties
        return out.to(dtype=torch.float32)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "dbn_raw": self.context.dbn_raw,
            "dbn_compact": self.context.dbn,
            "n_pairs": len(self.context.pairs),
            "n_crossing_pairs": len(self.context.crossing_pairs),
            "n_stack": len(self.context.stacks),
            "n_hairpin": len(self.context.hairpins),
            "n_bulge": len(self.context.bulges),
            "n_internal": len(self.context.internals),
            "skipped_pairs": self.context.skipped_pairs,
            "temperature_c": self.temperature_c,
            "top_k": self.top_k,
            "par_path": self.par_path,
        }


# -----------------------------------------------------------------------------
# External helpers / factories
# -----------------------------------------------------------------------------

def auto_parse_ss_for_pdb(pdb_path: str) -> str:
    if not HAS_ROUGH_SS:
        raise ImportError("[ERROR] Neither Src.coarse_grained_ss_parser nor Src.rough_ss_parser could be imported. Cannot use auto SS parsing.")

    abs_pdb = os.path.abspath(os.path.expanduser(pdb_path))
    residues = load_cg_residues_from_pdb(abs_pdb)
    n = len(residues)
    if n == 0:
        return ""

    cfg = RoughSSConfig(**AUTO_SS_PRESET)
    res = parse_cg_pdb_to_dbn(abs_pdb, cfg=cfg)
    return str(res.dbn_with_breaks)


def resolve_ss_arg_to_dbn(ss_arg: Optional[str], pdb_path: Optional[str] = None) -> str:
    if ss_arg is None:
        if pdb_path is None:
            raise ValueError("Either ss_arg or pdb_path must be provided")
        return auto_parse_ss_for_pdb(pdb_path)

    arg = str(ss_arg).strip()
    if arg.lower() == "auto":
        if pdb_path is None:
            raise ValueError("pdb_path is required when ss_arg='auto'")
        return auto_parse_ss_for_pdb(pdb_path)

    if _looks_like_dotbracket(arg):
        return arg

    if _load_ss_string_external is not None:
        try:
            raw = _load_ss_string_external(arg)
            if raw is not None and _looks_like_dotbracket(raw):
                return str(raw).strip()
        except Exception:
            pass

    path = os.path.abspath(os.path.expanduser(arg))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SS path not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return _extract_dotbracket_from_text(txt)


def build_mid_thermo_prior(
    ss_arg: Optional[str] = None,
    *,
    pdb_path: Optional[str] = None,
    temperature_c: float = 37.0,
    top_k: int = DEFAULT_TOP_K,
    unsupported_pair_penalty: float = 0.0,
    cache_limit: int = DEFAULT_CACHE_LIMIT,
    verbose: bool = False,
    par_path: Optional[str] = None,
) -> MidThermoPrior:
    return MidThermoPrior.from_ss_arg(
        ss_arg,
        pdb_path=pdb_path,
        temperature_c=temperature_c,
        top_k=top_k,
        unsupported_pair_penalty=unsupported_pair_penalty,
        cache_limit=cache_limit,
        verbose=verbose,
        par_path=par_path,
    )


build_seq_prior = build_mid_thermo_prior
build_thermo_ss = build_mid_thermo_prior


if __name__ == "__main__":  # pragma: no cover
    prior = MidThermoPrior(dbn="(((...)))")
    print(prior.summary())
    print(prior.score_one_seq_str("GGGAAACCC", return_breakdown=True))
