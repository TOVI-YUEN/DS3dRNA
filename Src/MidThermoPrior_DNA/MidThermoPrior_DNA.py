#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Primer3-config DNA local thermodynamic reranker for DS3dRNA.

The public API mirrors ``Src.MidThermoPrior`` so the design core can switch
between RNA and DNA without changing its scoring calls.  Unlike the RNA
implementation, this module reads Primer3's DNA ``*.dh``/``*.ds`` tables and
evaluates local motifs from

    delta_G(T) = (delta_H - T * delta_S) / 1000

where Primer3 stores enthalpy in cal/mol and entropy in cal/(mol K).

This remains a local mid-stage reranker rather than a full Primer3 ``thal``
dynamic-programming evaluator.  It scores the stack, hairpin, bulge, and
single-child internal-loop motifs implied by the supplied dot-bracket
structure, and skips multibranch/exterior/crossing bookkeeping in the same
way as the RNA implementation.
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from Src import MidThermoPrior as _rna
except Exception:  # pragma: no cover - supports direct execution in this folder
    import MidThermoPrior as _rna


DEFAULT_TOP_K = _rna.DEFAULT_TOP_K
DEFAULT_CACHE_LIMIT = _rna.DEFAULT_CACHE_LIMIT

IDX_TO_BASE_DNA = {0: "A", 1: "T", 2: "C", 3: "G"}
DNA_BASES = frozenset("ATCG")
DNA_CANONICAL_PAIRS = frozenset({"AT", "TA", "CG", "GC"})

# Constants used by Primer3's thal.c.
PRIMER3_AT_H = 2200.0
PRIMER3_AT_S = 6.9
PRIMER3_ILAS = -300.0 / 310.15
PRIMER3_ILAH = 0.0
PRIMER3_MAX_LOOP = 30

_REQUIRED_CONFIG_FILES = (
    "dangle.dh",
    "dangle.ds",
    "loops.dh",
    "loops.ds",
    "stack.dh",
    "stack.ds",
    "stackmm.dh",
    "stackmm.ds",
    "tetraloop.dh",
    "tetraloop.ds",
    "triloop.dh",
    "triloop.ds",
    "tstack.dh",
    "tstack2.dh",
    "tstack2.ds",
    "tstack_tm_inf.ds",
)


def _candidate_default_config_paths() -> List[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(here, "..", "primer3_config"),
        os.path.join(here, "primer3_config"),
        os.path.join(os.getcwd(), "Src", "primer3_config"),
        os.path.join(os.getcwd(), "primer3_config"),
    ]


def _is_primer3_config_dir(path: str) -> bool:
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, name))
        for name in _REQUIRED_CONFIG_FILES
    )


def _resolve_config_path(
    config_path: Optional[str] = None,
    par_path: Optional[str] = None,
) -> str:
    requested = config_path if config_path is not None else par_path
    if requested:
        path = os.path.abspath(os.path.expanduser(str(requested)))
        if not _is_primer3_config_dir(path):
            missing = [
                name for name in _REQUIRED_CONFIG_FILES
                if not os.path.isfile(os.path.join(path, name))
            ]
            raise FileNotFoundError(
                "Invalid Primer3 thermodynamic config directory: "
                f"{path}. Missing files: {', '.join(missing) if missing else '<not a directory>'}"
            )
        return path

    for candidate in _candidate_default_config_paths():
        path = os.path.abspath(candidate)
        if _is_primer3_config_dir(path):
            return path

    expected = os.path.abspath(_candidate_default_config_paths()[0])
    raise FileNotFoundError(
        "Primer3 thermodynamic config directory not found. "
        f"Expected it at {expected}, or pass config_path/par_path explicitly."
    )


def _parse_number(token: str) -> float:
    value = str(token).strip()
    if value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    if value.lower() in {"-inf", "-infinity"}:
        return float("-inf")
    return float(value)


def _dg_from_hs(param: Optional[Tuple[float, float]], temperature_k: float) -> Optional[float]:
    if param is None:
        return None
    dh, ds = float(param[0]), float(param[1])
    if not (math.isfinite(dh) and math.isfinite(ds)):
        return None
    return (dh - float(temperature_k) * ds) / 1000.0


class Primer3ConfigData:
    """Parser for the DNA thermodynamic files consumed by Primer3 ``thal``."""

    def __init__(self, config_path: str):
        self.config_path = _resolve_config_path(config_path=config_path)
        self.tables = self._load_all()

    def _path(self, filename: str) -> str:
        return os.path.join(self.config_path, filename)

    def _keyed_rows(self, filename: str) -> List[Tuple[str, float]]:
        rows: List[Tuple[str, float]] = []
        with open(self._path(filename), "r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    continue
                rows.append((fields[0].upper(), _parse_number(fields[1])))
        return rows

    def _paired_keyed_table(
        self,
        dh_filename: str,
        ds_filename: str,
    ) -> Dict[str, Tuple[float, float]]:
        dh_rows = self._keyed_rows(dh_filename)
        ds_rows = self._keyed_rows(ds_filename)
        if len(dh_rows) != len(ds_rows):
            raise ValueError(
                f"Primer3 table length mismatch: {dh_filename} has {len(dh_rows)} rows, "
                f"{ds_filename} has {len(ds_rows)} rows"
            )

        table: Dict[str, Tuple[float, float]] = {}
        for row_index, ((key_h, dh), (key_s, ds)) in enumerate(zip(dh_rows, ds_rows), start=1):
            if key_h != key_s:
                raise ValueError(
                    f"Primer3 table key mismatch at row {row_index}: "
                    f"{dh_filename}={key_h}, {ds_filename}={key_s}"
                )
            table[key_h] = (dh, ds)
        return table

    def _loops(self) -> Dict[str, Dict[int, Tuple[float, float]]]:
        def read(filename: str) -> Dict[int, Tuple[float, float, float]]:
            out: Dict[int, Tuple[float, float, float]] = {}
            with open(self._path(filename), "r", encoding="utf-8", errors="ignore") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split()
                    if len(fields) < 4:
                        continue
                    size = int(fields[0])
                    out[size] = tuple(_parse_number(x) for x in fields[1:4])
            return out

        dh_rows = read("loops.dh")
        ds_rows = read("loops.ds")
        if set(dh_rows) != set(ds_rows):
            raise ValueError("Primer3 loops.dh and loops.ds contain different loop sizes")

        names = ("interior", "bulge", "hairpin")
        result = {name: {} for name in names}
        for size in sorted(dh_rows):
            for column, name in enumerate(names):
                result[name][size] = (dh_rows[size][column], ds_rows[size][column])
        return result

    def _dangles(self) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
        dh_rows = self._keyed_rows("dangle.dh")
        ds_rows = self._keyed_rows("dangle.ds")
        if len(dh_rows) != len(ds_rows) or len(dh_rows) % 2 != 0:
            raise ValueError("Primer3 dangle tables must have equal, even row counts")

        combined: List[Tuple[str, Tuple[float, float]]] = []
        for (key_h, dh), (key_s, ds) in zip(dh_rows, ds_rows):
            if key_h != key_s:
                raise ValueError(f"Primer3 dangle key mismatch: {key_h} != {key_s}")
            combined.append((key_h, (dh, ds)))

        split = len(combined) // 2
        return dict(combined[:split]), dict(combined[split:])

    def _load_all(self) -> Dict[str, Any]:
        loops = self._loops()
        dangle3, dangle5 = self._dangles()
        return {
            "stack": self._paired_keyed_table("stack.dh", "stack.ds"),
            "stackmm": self._paired_keyed_table("stackmm.dh", "stackmm.ds"),
            # Primer3 deliberately pairs tstack.dh with tstack_tm_inf.ds.
            "tstack": self._paired_keyed_table("tstack.dh", "tstack_tm_inf.ds"),
            "tstack2": self._paired_keyed_table("tstack2.dh", "tstack2.ds"),
            "hairpin": loops["hairpin"],
            "bulge": loops["bulge"],
            "interior": loops["interior"],
            "triloop": self._paired_keyed_table("triloop.dh", "triloop.ds"),
            "tetraloop": self._paired_keyed_table("tetraloop.dh", "tetraloop.ds"),
            "dangle3": dangle3,
            "dangle5": dangle5,
            "terminal_at": (PRIMER3_AT_H, PRIMER3_AT_S),
        }

    def load_all_local(self) -> Dict[str, Any]:
        return self.tables


class MidThermoPrior(_rna.MidThermoPrior):
    """Primer3 DNA implementation with the RNA reranker's public interface."""

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
        config_path: Optional[str] = None,
    ) -> None:
        self.temperature_c = float(temperature_c)
        self.temperature_k = 273.15 + self.temperature_c
        self.top_k = int(top_k)
        self.unsupported_pair_penalty = float(unsupported_pair_penalty)
        self.cache_limit = int(max(0, cache_limit))
        self.verbose = bool(verbose)

        self.config_path = _resolve_config_path(config_path=config_path, par_path=par_path)
        # Keep the attribute used by the existing core's diagnostics.
        self.par_path = self.config_path
        self.primer3_config = Primer3ConfigData(self.config_path)
        self.tables = self.primer3_config.load_all_local()

        compact_dbn, seg_ids, _ = _rna._normalize_dbn_with_breaks(dbn)
        pair = _rna._pair_table_from_dbn(compact_dbn)
        self.context = self._build_context(dbn, compact_dbn, seg_ids, pair)
        self.length = len(self.context.dbn)
        self._score_cache: Dict[str, Dict[str, float]] = {}

        self.enable_fast_tensor = False
        self._dense_luts_by_device: Dict[str, Dict[str, Any]] = {}
        self._dense_lut_meta: Dict[str, Any] = {}
        self._fast_verify_samples = 0
        self._fast_verify_atol = 0.0
        self._fast_verify_rtol = 0.0
        self._fast_verify_warn_only = True

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
        config_path: Optional[str] = None,
    ) -> "MidThermoPrior":
        dbn = _rna.resolve_ss_arg_to_dbn(ss_arg=ss_arg, pdb_path=pdb_path)
        return cls(
            dbn=dbn,
            temperature_c=temperature_c,
            top_k=top_k,
            unsupported_pair_penalty=unsupported_pair_penalty,
            cache_limit=cache_limit,
            verbose=verbose,
            par_path=par_path,
            config_path=config_path,
        )

    def score_one_seq_str(self, seq: str, return_breakdown: bool = False):
        # Index 1 is U in DS3dRNA tensors but represents T in DNA mode.
        seq = str(seq).strip().upper().replace("U", "T")
        if len(seq) != self.length:
            raise ValueError(f"Sequence length mismatch: got {len(seq)}, expected {self.length}")
        if any(ch not in DNA_BASES for ch in seq):
            raise ValueError(f"Sequence contains unsupported DNA symbols: {seq!r}")

        if seq in self._score_cache:
            breakdown = self._score_cache[seq]
            return dict(breakdown) if return_breakdown else breakdown["total_dG"]

        breakdown = self._score_sequence(seq)
        if self.cache_limit > 0:
            if len(self._score_cache) >= self.cache_limit:
                try:
                    self._score_cache.pop(next(iter(self._score_cache)))
                except Exception:
                    self._score_cache.clear()
            self._score_cache[seq] = breakdown
        return dict(breakdown) if return_breakdown else breakdown["total_dG"]

    def _unsupported_penalty(self, pair_type: str) -> float:
        return 0.0 if pair_type in DNA_CANONICAL_PAIRS else self.unsupported_pair_penalty

    def _table_dg(self, table_name: str, key: str) -> Optional[float]:
        return _dg_from_hs(self.tables[table_name].get(key), self.temperature_k)

    def _loop_dg(self, table_name: str, size: int) -> float:
        table = self.tables[table_name]
        if not table:
            if self.verbose:
                print(f"[WARN][thermo-DNA] Primer3 loop table {table_name!r} is empty")
            return 0.0

        lookup_size = max(1, min(int(size), PRIMER3_MAX_LOOP))
        param = table.get(lookup_size)
        value = _dg_from_hs(param, self.temperature_k)
        if value is None:
            if self.verbose:
                print(
                    f"[WARN][thermo-DNA] non-finite {table_name} loop parameter "
                    f"for size={lookup_size}; using 0.0"
                )
            return 0.0
        return float(value)

    def _hairpin_initiation(self, n: int) -> float:
        return self._loop_dg("hairpin", n)

    def _bulge_initiation(self, n: int) -> float:
        return self._loop_dg("bulge", n)

    def _internal_initiation(self, n: int) -> float:
        return self._loop_dg("interior", n)

    def _internal_asymmetry(self, delta: int) -> float:
        if delta <= 0:
            return 0.0
        dh = PRIMER3_ILAH * float(delta)
        ds = PRIMER3_ILAS * float(delta)
        return (dh - self.temperature_k * ds) / 1000.0

    def _terminal_at_dg(self, pair: str) -> float:
        if pair not in {"AT", "TA"}:
            return 0.0
        value = _dg_from_hs(self.tables["terminal_at"], self.temperature_k)
        return 0.0 if value is None else float(value)

    def _score_stack(self, seq: str, motif: Any) -> Tuple[float, float]:
        key = (
            seq[motif.outer_i]
            + seq[motif.inner_i]
            + "_"
            + seq[motif.outer_j]
            + seq[motif.inner_j]
        )
        value = self._table_dg("stack", key)
        outer = seq[motif.outer_i] + seq[motif.outer_j]
        inner = seq[motif.inner_i] + seq[motif.inner_j]
        penalty = self._unsupported_penalty(outer) + self._unsupported_penalty(inner)
        return (0.0 if value is None else value), penalty

    def _score_hairpin(self, seq: str, motif: Any) -> Tuple[float, float]:
        i, j = motif.i, motif.j
        loop_size = j - i - 1
        if loop_size < 3:
            return 1e6, 0.0

        close = seq[i] + seq[j]
        value = self._hairpin_initiation(loop_size)

        if loop_size > 3:
            key = seq[i] + seq[i + 1] + "_" + seq[j] + seq[j - 1]
            terminal_mismatch = self._table_dg("tstack2", key)
            if terminal_mismatch is not None:
                value += terminal_mismatch
        elif loop_size == 3:
            value += self._terminal_at_dg(close)

        full_loop = seq[i : j + 1]
        if loop_size == 3:
            bonus = self._table_dg("triloop", full_loop)
            if bonus is not None:
                value += bonus
        elif loop_size == 4:
            bonus = self._table_dg("tetraloop", full_loop)
            if bonus is not None:
                value += bonus

        return value, self._unsupported_penalty(close)

    def _score_bulge(self, seq: str, motif: Any) -> Tuple[float, float]:
        loop_size = motif.left_unpaired + motif.right_unpaired
        outer = seq[motif.outer_i] + seq[motif.outer_j]
        inner = seq[motif.inner_i] + seq[motif.inner_j]
        penalty = self._unsupported_penalty(outer) + self._unsupported_penalty(inner)
        if loop_size <= 0:
            return 0.0, penalty
        if loop_size > PRIMER3_MAX_LOOP:
            # Primer3 thal rejects bulge/internal transitions above maxLoop.
            return 1e6, penalty

        value = self._bulge_initiation(loop_size)
        if loop_size == 1:
            key = (
                seq[motif.outer_i]
                + seq[motif.inner_i]
                + "_"
                + seq[motif.outer_j]
                + seq[motif.inner_j]
            )
            stack_value = self._table_dg("stack", key)
            if stack_value is not None:
                value += stack_value
        else:
            value += self._terminal_at_dg(outer)
            value += self._terminal_at_dg(inner)
        return value, penalty

    def _score_internal(self, seq: str, motif: Any) -> Tuple[float, float]:
        left_size = motif.left_unpaired
        right_size = motif.right_unpaired
        loop_size = left_size + right_size
        outer = seq[motif.outer_i] + seq[motif.outer_j]
        inner = seq[motif.inner_i] + seq[motif.inner_j]
        penalty = self._unsupported_penalty(outer) + self._unsupported_penalty(inner)
        if loop_size > PRIMER3_MAX_LOOP:
            return 1e6, penalty

        key_outer = (
            seq[motif.outer_i]
            + seq[motif.outer_i + 1]
            + "_"
            + seq[motif.outer_j]
            + seq[motif.outer_j - 1]
        )
        key_inner = (
            seq[motif.inner_j]
            + seq[motif.inner_j + 1]
            + "_"
            + seq[motif.inner_i]
            + seq[motif.inner_i - 1]
        )

        if left_size == 1 and right_size == 1:
            value = 0.0
            for key in (key_outer, key_inner):
                term = self._table_dg("stackmm", key)
                if term is not None:
                    value += term
            return value, penalty

        value = self._internal_initiation(loop_size)
        for key in (key_outer, key_inner):
            term = self._table_dg("tstack", key)
            if term is not None:
                value += term
        value += self._internal_asymmetry(abs(left_size - right_size))
        return value, penalty

    def _make_4d_lut(self, table_name: str, device: "torch.device") -> "torch.Tensor":
        lut = torch.full((4, 4, 4, 4), float("nan"), dtype=torch.float64, device=device)
        for a, b, c, d in itertools.product(range(4), repeat=4):
            key = IDX_TO_BASE_DNA[a] + IDX_TO_BASE_DNA[b] + "_" + IDX_TO_BASE_DNA[c] + IDX_TO_BASE_DNA[d]
            value = self._table_dg(table_name, key)
            if value is not None:
                lut[a, b, c, d] = float(value)
        return lut

    @staticmethod
    def _sequence_code(seq: str) -> int:
        base_to_idx = {base: idx for idx, base in IDX_TO_BASE_DNA.items()}
        code = 0
        for base in seq:
            code = code * 4 + base_to_idx[base]
        return code

    def _special_loop_lut(
        self,
        table_name: str,
        sequence_length: int,
        device: "torch.device",
    ) -> "torch.Tensor":
        lut = torch.full((4 ** sequence_length,), float("nan"), dtype=torch.float64, device=device)
        for seq, param in self.tables[table_name].items():
            if len(seq) != sequence_length or any(base not in DNA_BASES for base in seq):
                continue
            value = _dg_from_hs(param, self.temperature_k)
            if value is not None:
                lut[self._sequence_code(seq)] = float(value)
        return lut

    def _get_dense_luts_for_device(self, device: "torch.device") -> Dict[str, Any]:
        key = str(device)
        cached = self._dense_luts_by_device.get(key)
        if cached is not None:
            return cached

        dtype = torch.float64
        pair_penalty = torch.full(
            (4, 4),
            float(self.unsupported_pair_penalty),
            dtype=dtype,
            device=device,
        )
        for left, right in DNA_CANONICAL_PAIRS:
            base_to_idx = {base: idx for idx, base in IDX_TO_BASE_DNA.items()}
            pair_penalty[base_to_idx[left], base_to_idx[right]] = 0.0

        terminal_at = torch.zeros((4, 4), dtype=dtype, device=device)
        terminal_at_value = _dg_from_hs(self.tables["terminal_at"], self.temperature_k)
        if terminal_at_value is not None:
            terminal_at[0, 1] = float(terminal_at_value)
            terminal_at[1, 0] = float(terminal_at_value)

        luts: Dict[str, Any] = {
            "stack": self._make_4d_lut("stack", device),
            "stackmm": self._make_4d_lut("stackmm", device),
            "tstack": self._make_4d_lut("tstack", device),
            "tstack2": self._make_4d_lut("tstack2", device),
            "pair_penalty": pair_penalty,
            "terminal_at": terminal_at,
            "triloop": self._special_loop_lut("triloop", 5, device),
            "tetraloop": self._special_loop_lut("tetraloop", 6, device),
        }
        self._dense_luts_by_device[key] = luts
        return luts

    @staticmethod
    def _tensor_sequence_code(x: "torch.Tensor", positions: Sequence[int]) -> "torch.Tensor":
        code = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        for position in positions:
            code = code * 4 + x[:, position]
        return code

    def _score_batch_idx_torch_fast(self, candidate_batch: "torch.Tensor") -> "torch.Tensor":
        if candidate_batch.ndim != 2:
            raise ValueError(
                f"DNA fast tensor backend expects [B,N], got shape={tuple(candidate_batch.shape)}"
            )
        if int(candidate_batch.shape[1]) != self.length:
            raise ValueError(
                f"Sequence length mismatch: got {candidate_batch.shape[1]}, expected {self.length}"
            )
        x = candidate_batch.to(dtype=torch.long)
        if bool(torch.any((x < 0) | (x > 3)).item()):
            raise ValueError("DNA candidate tensor contains base indices outside 0..3")

        device = x.device
        luts = self._get_dense_luts_for_device(device)
        dtype = torch.float64
        batch_size = int(x.shape[0])
        scores = torch.zeros((batch_size,), dtype=dtype, device=device)
        penalties = torch.zeros((batch_size,), dtype=dtype, device=device)

        def add_table(table: "torch.Tensor", a: int, b: int, c: int, d: int) -> None:
            nonlocal scores
            values = table[x[:, a], x[:, b], x[:, c], x[:, d]]
            scores += torch.nan_to_num(values, nan=0.0)

        def add_pair_penalty(i: int, j: int) -> None:
            nonlocal penalties
            penalties += luts["pair_penalty"][x[:, i], x[:, j]]

        for motif in self.context.stacks:
            add_table(
                luts["stack"],
                motif.outer_i,
                motif.inner_i,
                motif.outer_j,
                motif.inner_j,
            )
            add_pair_penalty(motif.outer_i, motif.outer_j)
            add_pair_penalty(motif.inner_i, motif.inner_j)

        for motif in self.context.hairpins:
            i, j = motif.i, motif.j
            loop_size = j - i - 1
            if loop_size < 3:
                scores += 1e6
                continue

            scores += float(self._hairpin_initiation(loop_size))
            if loop_size > 3:
                add_table(luts["tstack2"], i, i + 1, j, j - 1)
            else:
                scores += luts["terminal_at"][x[:, i], x[:, j]]

            if loop_size == 3:
                code = self._tensor_sequence_code(x, range(i, j + 1))
                scores += torch.nan_to_num(luts["triloop"][code], nan=0.0)
            elif loop_size == 4:
                code = self._tensor_sequence_code(x, range(i, j + 1))
                scores += torch.nan_to_num(luts["tetraloop"][code], nan=0.0)

            add_pair_penalty(i, j)

        for motif in self.context.bulges:
            loop_size = motif.left_unpaired + motif.right_unpaired
            add_pair_penalty(motif.outer_i, motif.outer_j)
            add_pair_penalty(motif.inner_i, motif.inner_j)
            if loop_size <= 0:
                continue
            if loop_size > PRIMER3_MAX_LOOP:
                scores += 1e6
                continue

            scores += float(self._bulge_initiation(loop_size))
            if loop_size == 1:
                add_table(
                    luts["stack"],
                    motif.outer_i,
                    motif.inner_i,
                    motif.outer_j,
                    motif.inner_j,
                )
            else:
                scores += luts["terminal_at"][x[:, motif.outer_i], x[:, motif.outer_j]]
                scores += luts["terminal_at"][x[:, motif.inner_i], x[:, motif.inner_j]]

        for motif in self.context.internals:
            left_size = motif.left_unpaired
            right_size = motif.right_unpaired
            add_pair_penalty(motif.outer_i, motif.outer_j)
            add_pair_penalty(motif.inner_i, motif.inner_j)
            if left_size + right_size > PRIMER3_MAX_LOOP:
                scores += 1e6
                continue

            if left_size == 1 and right_size == 1:
                table = luts["stackmm"]
            else:
                scores += float(self._internal_initiation(left_size + right_size))
                scores += float(self._internal_asymmetry(abs(left_size - right_size)))
                table = luts["tstack"]

            add_table(
                table,
                motif.outer_i,
                motif.outer_i + 1,
                motif.outer_j,
                motif.outer_j - 1,
            )
            add_table(
                table,
                motif.inner_j,
                motif.inner_j + 1,
                motif.inner_i,
                motif.inner_i - 1,
            )

        return (scores + penalties).to(dtype=torch.float32)

    def summary(self) -> Dict[str, Any]:
        return {
            "backend": "primer3_config_DNA",
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
            "config_path": self.config_path,
        }


MidThermoPriorDNA = MidThermoPrior


def auto_parse_ss_for_pdb(pdb_path: str) -> str:
    return _rna.auto_parse_ss_for_pdb(pdb_path)


def resolve_ss_arg_to_dbn(ss_arg: Optional[str], pdb_path: Optional[str] = None) -> str:
    return _rna.resolve_ss_arg_to_dbn(ss_arg=ss_arg, pdb_path=pdb_path)


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
    config_path: Optional[str] = None,
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
        config_path=config_path,
    )


build_seq_prior = build_mid_thermo_prior
build_thermo_ss = build_mid_thermo_prior


__all__ = [
    "Primer3ConfigData",
    "MidThermoPrior",
    "MidThermoPriorDNA",
    "build_mid_thermo_prior",
    "build_seq_prior",
    "build_thermo_ss",
    "auto_parse_ss_for_pdb",
    "resolve_ss_arg_to_dbn",
]


if __name__ == "__main__":  # pragma: no cover
    prior = MidThermoPrior(dbn="(((...)))")
    print(prior.summary())
    print(prior.score_one_seq_str("GGGAAACCC", return_breakdown=True))
