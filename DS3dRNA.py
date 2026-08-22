#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command-line interface for DS3dRNA design and sequence ranking.

Design usage:
  1) Single PDB:
     python3 DS3dRNA.py ./one.pdb --batch 10 --temp 4.5

  2) Batch directory:
     python3 DS3dRNA.py ./pdb_dir --batch 10 --temp 4.5

  3) Multi-state folder:
     python3 DS3dRNA.py -m ./multi_folder --batch 10 --temp 4.5

  4) Reproduce one known seed:
     python3 DS3dRNA.py ./one.pdb --seed 363022884 --temp 4.5
     python3 DS3dRNA.py -m ./multi_folder --seed 363022884 --temp 4.5

  5) Replay one known seed batch:
     python3 DS3dRNA.py ./one.pdb --seed_batch ./seed_list.txt --temp 4.5
     python3 DS3dRNA.py -m ./multi_folder --seed_batch ./seed_list.txt --temp 4.5


Rank usage:
  6) Single-state rank:
     python3 DS3dRNA.py -rank --str ./x.pdb --fa ./seqs.fasta --rank_bs 10240

  7) Batch rank:
     python3 DS3dRNA.py -rank --batch_rank --str ./pdb_dir --fa ./seqs.fasta

  8) Multi-state rank:
     python3 DS3dRNA.py -rank --multi_rank --str ./multi_dir --fa ./seqs.fasta

Notes:
- SS behavior:
    * Design mode:
        - If --ss is omitted, auto-parsed contacts are used for thermodynamic scoring only.
        - If --ss auto is provided, auto-parsed contacts are used for thermodynamic scoring
          and as a hard base-pair constraint.
        - If --ss none is provided, all contacts are disabled (an all-unpaired DBN is used).
        - If the user provides --ss <dot-bracket/dbn/path>, that user input is used instead.
    * Rank mode:
        - Rank does NOT use thermodynamic reranking.
        - --ss only provides a hard secondary-structure constraint filter.
        - If --ss is omitted in rank mode, SS is disabled.
        - --ss auto parses contacts; --ss none represents zero base pairs.
- Frozen sequence behavior:
    * --frz accepts a mask/file/directory; A/U/C/G/T positions are locked and '-' positions remain designable.
    * --frz is a design-time hard constraint and is ignored in rank mode.
"""

import os
import re
import sys
import csv
import gc
import io
import time
import glob
import inspect
import secrets
import argparse
import traceback
import zipfile
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Callable

import torch
import numpy as np

from Src.potential import TriRNASP_Potential
from Src.ss_constraint import looks_like_dotbracket
import importlib.util

try:
    from Src.coarse_grained_ss_parser import (
        parse_cg_pdb_to_dbn,
        parse_cg_pdb_to_dbn_unknown_seq,
        RoughSSConfig,
        load_cg_residues_from_pdb,
        assign_breaks_and_fragments,
    )
    HAS_ROUGH_SS = True
    HAS_UNKNOWN_SEQ_SS = True
except ImportError:
    try:
        from Src.coarse_grained_ss_parser import (
            parse_cg_pdb_to_dbn,
            RoughSSConfig,
            load_cg_residues_from_pdb,
            assign_breaks_and_fragments,
        )
        HAS_ROUGH_SS = True
        HAS_UNKNOWN_SEQ_SS = False
        parse_cg_pdb_to_dbn_unknown_seq = None
    except ImportError:
        HAS_ROUGH_SS = False
        HAS_UNKNOWN_SEQ_SS = False
        parse_cg_pdb_to_dbn_unknown_seq = None


_TM_TAG_RE = re.compile(r"_TM[SF]?\d+(?:\.\d+)?", re.IGNORECASE)


def _has_tm_tag_pdb(path: str) -> bool:
    name = os.path.basename(path)
    return _TM_TAG_RE.search(name) is not None


def _sort_multistate_pdbs_target_first(pdb_paths: List[str]) -> List[str]:
    return sorted(
        pdb_paths,
        key=lambda p: (
            1 if _has_tm_tag_pdb(p) else 0,
            os.path.basename(p).lower(),
        )
    )


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODE_DIR = os.path.join(_THIS_DIR, "Mode")


def _resolve_dev_candidates() -> List[str]:
    return [
        os.path.join(MODE_DIR, "3dRNAdesign.py"),
        os.path.join(_THIS_DIR, "3dRNAdesign.py"),
    ]

def load_dev_core(mol_mode: str = "RNA"):
    dev_candidates = _resolve_dev_candidates()

    dev_path: Optional[str] = None
    for p in dev_candidates:
        if os.path.exists(p):
            dev_path = p
            break

    if dev_path is None:
        raise FileNotFoundError(
            "Cannot find computational core. Tried: "
            + ", ".join(x for x in dev_candidates)
        )

    spec = importlib.util.spec_from_file_location("dev3dRNAdesign", dev_path)
    dev_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Select the molecule-specific thermodynamic backend during module loading
    # without changing process-global environment state.
    dev_mod.DS3DRNA_MOL_MODE = str(mol_mode).strip().upper()
    # Register before execution for dataclass and runtime-introspection support.
    sys.modules[spec.name] = dev_mod
    spec.loader.exec_module(dev_mod)

    dev_ver = getattr(dev_mod, "__VERSION__", "unknown")
    print(f"[INFO] Loaded computational core: {dev_path} (version={dev_ver})")
    return dev_mod, dev_path


dev = None
DEV_PATH: Optional[str] = None

print_trirnade_banner = None
set_seed = None
design_one_pdb_single = None
write_top10_energy_csv = None
run_rank_mode = None
run_rank_batch = None
run_rank_multi = None
_default_rank_out_for_pdb = None
design_one_pdb_single_multi = None
design_multi_state_multi = None

IDX_TO_BASE = {0: "A", 1: "U", 2: "C", 3: "G"}

AUTO_SS_PRESET = {
    "wc_nn_min": 8.0,
    "wc_nn_max": 9.5,
    "wc_nn_ideal": 8.84,

    "nonwc_nn_min": 7.5,
    "nonwc_nn_max": 9.5,

    "min_score_gap": 0.2,
    "allow_gu_wobble": True,
    "allow_pseudoknot": True,

    "selection_mode": "original",
    "keep_isolated_pairs": True,

    "unknown_same_class_penalty": 0.0,
    "unknown_missing_class_penalty": 0.0,

    "enable_pk_sandwich_cleanup": True,
    "pk_cleanup_iters": 2,
    "pk_sandwich_min_pk_pairs": 4,
    "pk_sandwich_max_inner_pairs": 2,
    "pk_sandwich_min_ratio": 2.0,

    "use_visibility_competition": True,
    "visibility_dot_min": 0.5,
    "visibility_missing_is_visible": False,

    "enable_stem_gap_fill": True,
    "stem_gap_fill_score_max": None,
    "stem_gap_fill_require_continuity": False,
    "stem_gap_fill_flank_max_run": 1,

    "repair_require_preserve_existing_levels": True,
}

_AUTO_SS_CACHE: Dict[str, Dict[str, Any]] = {}

AUTO_SS_MODE_NAME = "unknown-seq-deterministic"
# ============================================================
# Smart SS Parser Invocation
# ============================================================

def auto_parse_ss_for_pdb(pdb_path: str) -> str:
    if not HAS_ROUGH_SS:
        raise ImportError("[ERROR] Src.coarse_grained_ss_parser could not be imported. Cannot use --ss auto.")

    abs_pdb = os.path.abspath(os.path.expanduser(pdb_path))
    residues = load_cg_residues_from_pdb(abs_pdb)
    n = len(residues)

    if n == 0:
        _AUTO_SS_CACHE[abs_pdb] = {
            "pdb_path": abs_pdb,
            "pdb_name": os.path.basename(abs_pdb),
            "nts": 0,
            "preset_name": AUTO_SS_MODE_NAME if HAS_UNKNOWN_SEQ_SS else "conservative-native-seq-fallback",
            "dbn": "",
            "params": AUTO_SS_PRESET.copy(),
            "auto_mode": "unknown_seq" if HAS_UNKNOWN_SEQ_SS else "native_seq_fallback",
            "pair_count": 0,
        }
        return ""

    cfg = RoughSSConfig(**AUTO_SS_PRESET)

    if HAS_UNKNOWN_SEQ_SS:

        res = parse_cg_pdb_to_dbn_unknown_seq(abs_pdb, cfg=cfg)
        preset_name = AUTO_SS_MODE_NAME
        auto_mode = "unknown_seq"
    else:

        print("[WARN][ss] unknown-seq parser unavailable; falling back to seq-aware parser.")
        res = parse_cg_pdb_to_dbn(abs_pdb, cfg=cfg)
        preset_name = "conservative-native-seq-fallback"
        auto_mode = "native_seq_fallback"

    pair_count = len(getattr(res, "kept_pairs", []) or [])

    _AUTO_SS_CACHE[abs_pdb] = {
        "pdb_path": abs_pdb,
        "pdb_name": os.path.basename(abs_pdb),
        "nts": n,
        "preset_name": preset_name,
        "dbn": res.dbn_with_breaks,
        "params": AUTO_SS_PRESET.copy(),
        "auto_mode": auto_mode,
        "pair_count": pair_count,
    }

    return res.dbn_with_breaks


def no_contact_ss_for_pdb(pdb_path: str) -> str:
    """Build an all-unpaired DBN while preserving inferred chain breaks."""
    if not HAS_ROUGH_SS:
        raise ImportError(
            "[ERROR] Src.coarse_grained_ss_parser could not be imported. "
            "Cannot use --ss none."
        )

    abs_pdb = os.path.abspath(os.path.expanduser(pdb_path))
    residues = load_cg_residues_from_pdb(abs_pdb)
    break_after = assign_breaks_and_fragments(residues, RoughSSConfig(**AUTO_SS_PRESET))

    chars: List[str] = []
    for i in range(len(residues)):
        chars.append(".")
        if i < len(residues) - 1 and break_after[i]:
            chars.append("&")
    return "".join(chars)


# ============================================================
# Utilities
# ============================================================

def seq_idx_to_masked_str_local(seq_idx, native_idx, mol: str = "RNA", unknown_char: str = "-") -> str:
    mol = normalize_mol(mol)

    if isinstance(seq_idx, torch.Tensor):
        seq = seq_idx.detach().cpu().tolist()
    else:
        seq = list(seq_idx)

    if isinstance(native_idx, torch.Tensor):
        nat = native_idx.detach().cpu().tolist()
    else:
        nat = list(native_idx)

    if len(seq) != len(nat):
        raise ValueError(f"length mismatch: seq={len(seq)} native={len(nat)}")

    if mol == "DNA":
        base_map = {0: "A", 1: "T", 2: "C", 3: "G"}
    else:
        base_map = {0: "A", 1: "U", 2: "C", 3: "G"}

    out = []
    for s, n in zip(seq, nat):
        if int(n) < 0:
            out.append(unknown_char)
        else:
            s = int(s)
            out.append(base_map.get(s, "N"))
    return "".join(out)


def has_masked_sites(native_idx) -> bool:
    if native_idx is None:
        return False
    if isinstance(native_idx, torch.Tensor):
        return bool((native_idx < 0).any().item())
    arr = np.asarray(native_idx)
    return bool((arr < 0).any())


def count_masked_sites(native_idx) -> int:
    if native_idx is None:
        return 0
    if isinstance(native_idx, torch.Tensor):
        return int((native_idx < 0).sum().item())
    arr = np.asarray(native_idx)
    return int((arr < 0).sum())


DEFAULT_T_MIN = 4.5
ENSEMBLE_BETA = 0.0
SS_STEPS = 10000
SS_TAIL_STEPS = 4000
SS_TOPE = 10000


def compute_E_fine_DS(E_fine: float, Ensemble_E: float, E_rough: float) -> float:
    return (float(E_fine) + 0.0 * float(Ensemble_E) + 0.0 * float(E_rough)) / 1.0


def normalize_mol(mol: str) -> str:
    mol = str(mol).strip().upper()
    if mol not in {"RNA", "DNA"}:
        raise ValueError(f"[ERROR] --mol must be RNA or DNA, got: {mol}")
    return mol


def _looks_like_output_dotbracket(text: str) -> bool:
    compact = "".join(ch for ch in str(text) if not ch.isspace())
    if not compact:
        return False
    allowed = set(".()[]{}<>&")
    return all(ch in allowed for ch in compact) and any(ch in ".()[]{}<>" for ch in compact)


def _extract_output_dotbracket_from_text(text: str) -> Optional[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">") or line.startswith("#") or line.startswith(";"):
            continue
        if looks_like_dotbracket(line) or _looks_like_output_dotbracket(line):
            return line
    return None


def _dotbracket_from_ss_arg(ss_arg: Optional[str]) -> Optional[str]:
    if ss_arg is None:
        return None

    raw = str(ss_arg).strip()
    if not raw:
        return None

    if looks_like_dotbracket(raw) or _looks_like_output_dotbracket(raw):
        return raw

    path = os.path.abspath(os.path.expanduser(raw))
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return _extract_output_dotbracket_from_text(f.read())
        except Exception as e:
            print(f"[WARN][ss] Failed to read chain-break SS file: {path} | {e}")
            return None

    return None


def chain_break_info_from_ss(ss_arg: Optional[str]) -> Optional[Tuple[List[int], int]]:
    dbn = _dotbracket_from_ss_arg(ss_arg)
    if not dbn or "&" not in dbn:
        return None

    breaks: List[int] = []
    compact_len = 0
    for ch in dbn:
        if ch.isspace():
            continue
        if ch == "&":
            if compact_len > 0 and (not breaks or breaks[-1] != compact_len):
                breaks.append(compact_len)
            continue
        compact_len += 1

    breaks = [b for b in breaks if 0 < b < compact_len]
    if not breaks:
        return None
    return breaks, compact_len


def apply_chain_breaks_to_sequence(seq: str, chain_break_info: Optional[Tuple[List[int], int]] = None) -> str:
    if seq is None or chain_break_info is None:
        return seq

    breaks, compact_len = chain_break_info
    compact_seq = str(seq).replace("&", "")
    if len(compact_seq) != int(compact_len):
        return str(seq)

    break_set = set(int(x) for x in breaks)
    out = []
    for i, ch in enumerate(compact_seq):
        if i in break_set:
            out.append("&")
        out.append(ch)
    return "".join(out)


def seq_display(seq: str, mol: str, chain_break_info: Optional[Tuple[List[int], int]] = None) -> str:
    if seq is None:
        return seq
    mol = normalize_mol(mol)
    if mol == "DNA":
        seq = str(seq).replace("U", "T")
    else:
        seq = str(seq)
    return apply_chain_breaks_to_sequence(seq, chain_break_info)


def postprocess_rank_csv_sequences(
    out_csv: str,
    mol: str,
    chain_break_info: Optional[Tuple[List[int], int]] = None,
):
    if not out_csv:
        return
    p = os.path.abspath(os.path.expanduser(out_csv))
    if not os.path.isfile(p):
        return

    tmp_path = None
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            if not fieldnames or "sequence" not in fieldnames:
                return
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=os.path.dirname(p) or ".",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
                writer = csv.DictWriter(tmp, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                changed = False
                for row in reader:
                    old_seq = row.get("sequence", "")
                    new_seq = seq_display(old_seq, mol, chain_break_info)
                    if new_seq != old_seq:
                        changed = True
                    row["sequence"] = new_seq
                    writer.writerow(row)

        if changed and tmp_path is not None:
            os.replace(tmp_path, p)
            tmp_path = None
    except Exception as e:
        print(f"[WARN] Failed rank sequence postprocess for CSV: {p} | {e}")
    finally:
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def csv_replace_u_with_t(path: str):
    if not path:
        return
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            txt = f.read()
        txt2 = txt.replace("U", "T")
        if txt2 != txt:
            with open(p, "w", encoding="utf-8") as f:
                f.write(txt2)
    except Exception as e:
        print(f"[WARN] Failed U->T postprocess for CSV: {p} | {e}")


def postprocess_rank_outputs_for_dna_single(out_csv: str):
    csv_replace_u_with_t(out_csv)


def postprocess_rank_outputs_for_dna_batch(rank_out: Optional[str], pdb_dir: str):
    candidates: List[str] = []

    if rank_out:
        ro = os.path.abspath(os.path.expanduser(rank_out))
        if os.path.isfile(ro) and ro.lower().endswith(".csv"):
            candidates.append(ro)
        elif os.path.isdir(ro):
            for p in sorted(glob.glob(os.path.join(ro, "*.csv"))):
                candidates.append(p)

    if not candidates:
        base_dir = os.path.abspath(os.path.expanduser(pdb_dir))
        for p in sorted(glob.glob(os.path.join(base_dir, "*.csv"))):
            candidates.append(p)

    for p in candidates:
        csv_replace_u_with_t(p)


def postprocess_rank_outputs_for_dna_multi(rank_out: Optional[str], multi_dir: str):
    candidates: List[str] = []

    if rank_out:
        ro = os.path.abspath(os.path.expanduser(rank_out))
        if os.path.isfile(ro) and ro.lower().endswith(".csv"):
            candidates.append(ro)
        elif os.path.isdir(ro):
            for p in sorted(glob.glob(os.path.join(ro, "*.csv"))):
                candidates.append(p)

    if not candidates:
        base_dir = os.path.abspath(os.path.expanduser(multi_dir))
        for p in sorted(glob.glob(os.path.join(base_dir, "*.csv"))):
            candidates.append(p)

    for p in candidates:
        csv_replace_u_with_t(p)


def rand_seed_32bit() -> int:
    return secrets.randbelow(2**31 - 1)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pdb_base_name(pdb_path: str) -> str:
    b = os.path.basename(pdb_path)
    return os.path.splitext(b)[0]


def resolve_energy_file(energy_root: str, stem: str) -> str:
    candidates = [
        os.path.join(energy_root, f"{stem}.npy"),
        os.path.join(energy_root, f"{stem}.energy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    tried = ", ".join(candidates)
    raise FileNotFoundError(f"[ERROR] {stem} energy file not found. Tried: {tried}")


def _summary_count(value: Any) -> int:
    try:
        return max(1, int(value))
    except Exception:
        return 1


def _summary_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("inf")


def _summary_seed_string(*values: Any) -> str:
    seen = set()
    merged: List[str] = []
    for value in values:
        for part in str(value or "").split(";"):
            seed = part.strip()
            if seed and seed not in seen:
                seen.add(seed)
                merged.append(seed)
    return ";".join(merged)


def _coerce_summary_row(row: Dict[str, Any], header: List[str]) -> Dict[str, str]:
    out = {h: str(row.get(h, "") if row.get(h, "") is not None else "") for h in header}
    if not out.get("Designed_seq"):
        out["Designed_seq"] = str(row.get("consensus_sequence", "") or "")
    if not out.get("count"):
        out["count"] = str(_summary_count(row.get("count", 1)))
    return out


def write_or_merge_main_csv(csv_path: str, row: List[Any], header: List[str]):
    ensure_dir(os.path.dirname(os.path.abspath(csv_path)) or ".")

    raw_rows: List[Dict[str, Any]] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            raw_rows.extend(reader)

    raw_rows.append({h: row[i] if i < len(row) else "" for i, h in enumerate(header)})

    merged_by_seq: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for raw in raw_rows:
        current = _coerce_summary_row(raw, header)
        seq = current.get("Designed_seq", "")
        if not seq:
            continue

        if seq not in merged_by_seq:
            current["count"] = str(_summary_count(current.get("count", 1)))
            merged_by_seq[seq] = current
            order.append(seq)
            continue

        existing = merged_by_seq[seq]
        merged_seed = _summary_seed_string(existing.get("seed", ""), current.get("seed", ""))
        merged_count = _summary_count(existing.get("count", 1)) + _summary_count(current.get("count", 1))

        if _summary_float(current.get("E_fine(kBT)", "")) < _summary_float(existing.get("E_fine(kBT)", "")):
            for h in header:
                if h not in {"Designed_seq", "count"}:
                    existing[h] = current.get(h, "")

        existing["seed"] = merged_seed
        existing["count"] = str(merged_count)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for seq in order:
            row_out = merged_by_seq[seq]
            w.writerow([row_out.get(h, "") for h in header])


def pretty_print_run(
    name: str,
    seed: int,
    designed_seq: str,
    rec: float,
    f1: float,
    ppl: float,
    div: float,
    E_fine_DS: float,
    topE: int,
    out_dir: str,
):
    print("------------------------------------------------------------")
    print(f"[DONE] {name}")
    print(f"  Seed             : {seed}")
    print(f"  Designed_seq     : {designed_seq}")
    print(f"  E_fine(kBT)      : {E_fine_DS:.6f}")
    print(f"  Recovery         : {rec:.6f}")
    print(f"  MacroF1          : {f1:.6f}")
    print(f"  PPL              : {ppl:.6f}")
    print(f"  Div_ensemble     : {div:.6f}")
    print(f"  Output dir       : {out_dir}")
    print(f"[INFO] trajTopE={topE} unique-seq records have been written as compressed CSV.")
    print("------------------------------------------------------------")


def _fail_log_path(out_dir: str) -> str:
    return os.path.join(out_dir, "fail.log")


def append_fail_log(
    out_dir: str,
    target_tag: str,
    seed: Optional[int],
    stage: str,
    exc: BaseException,
):
    ensure_dir(out_dir)
    p = _fail_log_path(out_dir)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tb = traceback.format_exc()

    with open(p, "a", encoding="utf-8") as f:
        f.write("============================================================\n")
        f.write(f"[TIME]   {ts}\n")
        f.write(f"[TARGET] {target_tag}\n")
        f.write(f"[SEED]   {'' if seed is None else seed}\n")
        f.write(f"[STAGE]  {stage}\n")
        f.write(f"[EXC]    {type(exc).__name__}: {exc}\n")
        f.write("[TRACEBACK]\n")
        f.write(tb)
        if not tb.endswith("\n"):
            f.write("\n")
        f.write("============================================================\n\n")


def read_seed_batch_file(path: str) -> List[int]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[ERROR] --seed_batch file not found: {path}")

    seeds: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            line = line.replace(",", " ")
            parts = [x for x in line.split() if x]
            for p in parts:
                try:
                    s = int(p)
                except Exception as e:
                    raise ValueError(f"[ERROR] Invalid seed in {path}: {p}") from e
                if s < 0:
                    raise ValueError(f"[ERROR] Seed must be non-negative, got: {s}")
                seeds.append(s)

    if len(seeds) == 0:
        raise ValueError(f"[ERROR] No valid seeds found in: {path}")
    return seeds


def write_seed_batch_file(
    out_dir: str,
    target_tag: str,
    round_idx: int,
    seeds: List[int],
) -> str:
    seed_dir = os.path.join(out_dir, "seed_batches")
    ensure_dir(seed_dir)

    now = datetime.now()
    ts_code = now.strftime("%Y%m%d_%H%M%S")
    base_name = f"seed_batch_{ts_code}"
    out_path = os.path.join(seed_dir, f"{base_name}.txt")

    if os.path.exists(out_path):
        idx = 1
        while True:
            cand = os.path.join(seed_dir, f"{base_name}_{idx:02d}.txt")
            if not os.path.exists(cand):
                out_path = cand
                break
            idx += 1

    human_ts = now.strftime("%Y-%m-%d %H:%M:%S")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# target: {target_tag}\n")
        f.write(f"# time:   {human_ts}\n")
        f.write(f"# round:  {round_idx}\n")
        f.write(f"# count:  {len(seeds)}\n")
        for s in seeds:
            f.write(f"{int(s)}\n")

    print(f"[INFO] Saved seed batch: {out_path}")
    return out_path


def _is_oom_error(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    msg = str(e).lower()
    return ("out of memory" in msg) or ("cuda out of memory" in msg) or ("cublas" in msg and "alloc" in msg)


def _cleanup_cuda():
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def _cleanup_after_oom():
    for _ in range(3):
        try:
            gc.collect()
        except Exception:
            pass

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        try:
            gc.collect()
        except Exception:
            pass


def _hard_cleanup_sleep(sleep_s: float = 0.5):
    _cleanup_after_oom()
    if sleep_s > 0:
        time.sleep(float(sleep_s))
    _cleanup_after_oom()


def _parse_dev_return_any(ret: Tuple[Any, ...], *, kind: str) -> Dict[str, Any]:
    if not isinstance(ret, tuple):
        ret = tuple(ret)

    if len(ret) >= 26:
        native_idx = ret[14]
        if isinstance(native_idx, torch.Tensor):
            native_idx = native_idx.detach().cpu().clone()
        elif native_idx is not None:
            native_idx = torch.as_tensor(native_idx, dtype=torch.long)

        return {
            "consensus_seq": ret[0],
            "rec": float(ret[1]),
            "f1": float(ret[2]),
            "ppl": float(ret[3]),
            "div": float(ret[4]),
            "E_rough": float(ret[5]),
            "E_cons": float(ret[6]),
            "traj_step": ret[17],
            "traj_rec": ret[18],
            "traj_f1": ret[19],
            "traj_seq_u8": ret[20],
            "traj_E": ret[21],
            "tail_rec": ret[22],
            "tail_f1": ret[23],
            "tail_seq_u8": ret[24],
            "tail_E": ret[25],
            "native_idx": native_idx,
            "native_seq_str": ret[15],
            "n_valid_sites": int(ret[16]),
        }

    # Fallback for older / alternate dev schemas.
    if len(ret) < 23:
        raise RuntimeError(f"[ERROR] developer return too short ({kind}): len={len(ret)}")

    tail_rec, tail_f1, tail_seq_u8, tail_E = ret[-4], ret[-3], ret[-2], ret[-1]
    traj_step, traj_rec, traj_f1, traj_seq_u8, traj_E = ret[-9], ret[-8], ret[-7], ret[-6], ret[-5]

    parsed = {
        "consensus_seq": ret[0],
        "rec": float(ret[1]),
        "f1": float(ret[2]),
        "ppl": float(ret[3]),
        "div": float(ret[4]),
        "E_rough": float(ret[5]),
        "E_cons": float(ret[6]),
        "traj_step": traj_step,
        "traj_rec": traj_rec,
        "traj_f1": traj_f1,
        "traj_seq_u8": traj_seq_u8,
        "traj_E": traj_E,
        "tail_rec": tail_rec,
        "tail_f1": tail_f1,
        "tail_seq_u8": tail_seq_u8,
        "tail_E": tail_E,
        "native_idx": None,
        "native_seq_str": None,
        "n_valid_sites": None,
    }

    extras = list(ret[7:])

    for x in extras:
        if isinstance(x, torch.Tensor):
            if x.dim() == 1 and x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long):
                vals = x.detach().cpu()
                if parsed["native_idx"] is None and ((vals >= -1).all().item() and (vals <= 3).all().item()):
                    parsed["native_idx"] = vals.clone()
                    continue

        if isinstance(x, (np.ndarray, list, tuple)):
            try:
                arr = np.asarray(x)
                if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
                    if parsed["native_idx"] is None and np.all(arr >= -1) and np.all(arr <= 3):
                        parsed["native_idx"] = torch.as_tensor(arr, dtype=torch.long)
                        continue
            except Exception:
                pass

        if isinstance(x, str):
            if parsed["native_seq_str"] is None and ("-" in x or set(x).issubset(set("AUCGTN-"))):
                parsed["native_seq_str"] = x
                continue

        if isinstance(x, (int, np.integer)):
            if parsed["n_valid_sites"] is None:
                parsed["n_valid_sites"] = int(x)
                continue

    return parsed


def _parse_dev_single_return(ret: Tuple[Any, ...]) -> Dict[str, Any]:
    return _parse_dev_return_any(ret, kind="single")


def _parse_dev_multi_return(ret: Tuple[Any, ...]) -> Dict[str, Any]:
    return _parse_dev_return_any(ret, kind="multi")


def _safe_len(x: Any) -> int:
    try:
        return 0 if x is None else len(x)
    except Exception:
        return 0


def _debug_print(res: Dict[str, Any], tag: str):
    def _t(x):
        return "None" if x is None else str(type(x)).replace("<class '", "").replace("'>", "")
    print(f"[DEBUG:{tag}] tail_seq_u8 type={_t(res.get('tail_seq_u8'))} len={_safe_len(res.get('tail_seq_u8'))}")
    print(f"[DEBUG:{tag}] tail_E      type={_t(res.get('tail_E'))} len={_safe_len(res.get('tail_E'))}")
    print(f"[DEBUG:{tag}] traj_seq_u8 type={_t(res.get('traj_seq_u8'))} len={_safe_len(res.get('traj_seq_u8'))}")
    print(f"[DEBUG:{tag}] traj_E      type={_t(res.get('traj_E'))} len={_safe_len(res.get('traj_E'))}")
    print(f"[DEBUG:{tag}] traj_step   type={_t(res.get('traj_step'))} len={_safe_len(res.get('traj_step'))}")


def topk_unique_by_min_energy_numpy(
    traj_seq_u8_list,
    traj_E_list,
    traj_step_list=None,
    traj_rec_list=None,
    traj_f1_list=None,
    k: int = 1000,
    reconstruct_step_if_missing: bool = True,
    step_start: int = 1,
    mol: str = "RNA",
    native_idx=None,
    chain_break_info: Optional[Tuple[List[int], int]] = None,
):
    k = max(1, int(k))
    M = len(traj_seq_u8_list)
    if M == 0:
        return []

    if isinstance(traj_seq_u8_list[0], np.ndarray):
        X = np.stack([s.astype(np.uint8, copy=False) for s in traj_seq_u8_list], axis=0)
    elif isinstance(traj_seq_u8_list[0], torch.Tensor):
        X = torch.stack([s.to(dtype=torch.uint8) for s in traj_seq_u8_list], dim=0).cpu().numpy()
    else:
        X = np.stack([np.asarray(s, dtype=np.uint8) for s in traj_seq_u8_list], axis=0)

    E = np.asarray(traj_E_list, dtype=np.float64)
    if E.shape[0] != M:
        raise RuntimeError(f"traj_E length mismatch: {E.shape[0]} vs {M}")

    if traj_step_list is None or len(traj_step_list) != M:
        if reconstruct_step_if_missing:
            steps = np.arange(step_start, step_start + M, dtype=np.int32)
        else:
            steps = np.full((M,), -1, dtype=np.int32)
    else:
        steps = np.asarray(traj_step_list, dtype=np.int32)

    rec = None
    f1 = None
    if traj_rec_list is not None and len(traj_rec_list) == M:
        rec = np.asarray(traj_rec_list, dtype=np.float64)
    if traj_f1_list is not None and len(traj_f1_list) == M:
        f1 = np.asarray(traj_f1_list, dtype=np.float64)

    N = X.shape[1]
    v = np.ascontiguousarray(X).view(np.dtype((np.void, X.dtype.itemsize * N))).reshape(-1)
    uniq_v, inv = np.unique(v, return_inverse=True)
    U = uniq_v.shape[0]

    cnt = np.bincount(inv, minlength=U).astype(np.int64)

    minE = np.full((U,), np.inf, dtype=np.float64)
    np.minimum.at(minE, inv, E)

    is_min = (E == minE[inv])
    step_min = np.full((U,), np.iinfo(np.int32).max, dtype=np.int32)
    np.minimum.at(step_min, inv[is_min], steps[is_min])

    rep_idx = None
    rec_min = None
    f1_min = None

    if rec is not None or f1 is not None:
        rep_idx = np.full((U,), -1, dtype=np.int64)
        idxs = np.nonzero(is_min)[0]
        for i in idxs:
            g = inv[i]
            if rep_idx[g] < 0:
                rep_idx[g] = i
        if rec is not None:
            rec_min = np.where(rep_idx >= 0, rec[rep_idx], np.nan)
        if f1 is not None:
            f1_min = np.where(rep_idx >= 0, f1[rep_idx], np.nan)

    top_idx = np.argsort(minE)[: min(k, U)]

    Xuniq = uniq_v.view(np.uint8).reshape(-1, N)

    records = []
    for g in top_idx:
        seq_u8 = Xuniq[g].copy()

        if native_idx is not None:
            seq_str = seq_idx_to_masked_str_local(seq_u8, native_idx, mol=mol, unknown_char="-")
        else:
            mol = normalize_mol(mol)
            if mol == "DNA":
                base_map = np.array(["A", "T", "C", "G"], dtype="<U1")
            else:
                base_map = np.array(["A", "U", "C", "G"], dtype="<U1")
            seq_str = "".join(base_map[seq_u8].tolist())
        seq_str = seq_display(seq_str, mol, chain_break_info)
        records.append({
            "seq_u8": seq_u8,
            "seq": seq_str,
            "E_min": float(minE[g]),
            "step_min": int(step_min[g]) if step_min[g] != np.iinfo(np.int32).max else -1,
            "rec_at_min": None if rec_min is None else (None if not np.isfinite(rec_min[g]) else float(rec_min[g])),
            "f1_at_min": None if f1_min is None else (None if not np.isfinite(f1_min[g]) else float(f1_min[g])),
            "count": int(cnt[g]),
        })
    return records


def _write_csv_zip(out_zip: str, inner_csv_name: str, header: List[str], rows: List[List[Any]]):
    ensure_dir(os.path.dirname(os.path.abspath(out_zip)) or ".")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open(inner_csv_name, "w") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)


def write_topE_traj_csv(out_zip: str, inner_csv_name: str, records: List[Dict[str, Any]]):
    header = ["rank", "step_min", "E_min(kBT)", "rec_at_min", "f1_at_min", "count", "sequence"]
    rows = []
    for i, r in enumerate(records, start=1):
        rows.append([
            i,
            r.get("step_min", -1),
            f"{r.get('E_min', float('nan')):.6f}",
            ("" if r.get("rec_at_min", None) is None else f"{r['rec_at_min']:.6f}"),
            ("" if r.get("f1_at_min", None) is None else f"{r['f1_at_min']:.6f}"),
            r.get("count", 1),
            r.get("seq", ""),
        ])
    _write_csv_zip(out_zip, inner_csv_name, header, rows)


def write_ensemble_csv(out_zip: str, inner_csv_name: str, records: List[Dict[str, Any]]):
    header = ["rank", "step_tail_min", "E_min(kBT)", "rec_at_min", "f1_at_min", "count", "sequence"]
    rows = []
    for i, r in enumerate(records, start=1):
        rows.append([
            i,
            r.get("step_min", -1),
            f"{r.get('E_min', float('nan')):.6f}",
            ("" if r.get("rec_at_min", None) is None else f"{r['rec_at_min']:.6f}"),
            ("" if r.get("f1_at_min", None) is None else f"{r['f1_at_min']:.6f}"),
            r.get("count", 1),
            r.get("seq", ""),
        ])
    _write_csv_zip(out_zip, inner_csv_name, header, rows)


def _export_topE_for_one_seed(
    out_dir: str,
    topE: int,
    seed: int,
    res: Dict[str, Any],
    mol: str,
    chain_break_info: Optional[Tuple[List[int], int]] = None,
):
    if res.get("traj_seq_u8") is not None and res.get("traj_E") is not None and \
       _safe_len(res.get("traj_seq_u8")) and _safe_len(res.get("traj_E")):

        top_records = topk_unique_by_min_energy_numpy(
            traj_step_list=res.get("traj_step"),
            traj_seq_u8_list=res.get("traj_seq_u8"),
            traj_E_list=res.get("traj_E"),
            traj_rec_list=res.get("traj_rec"),
            traj_f1_list=res.get("traj_f1"),
            k=topE,
            reconstruct_step_if_missing=True,
            step_start=1,
            mol=mol,
            native_idx=res.get("native_idx"),
            chain_break_info=chain_break_info,
        )
        inner_csv = f"trajTopE_{topE}_{seed}.csv"
        top_zip = os.path.join(out_dir, f"{inner_csv}.zip")
        write_topE_traj_csv(top_zip, inner_csv, top_records)
        return

    print("[WARN][tailTopE] traj_* missing from computational core; exporting from tail_* (step_min reconstructed).")
    top_records = topk_unique_by_min_energy_numpy(
        traj_step_list=None,
        traj_seq_u8_list=res.get("tail_seq_u8") or [],
        traj_E_list=res.get("tail_E") or [],
        traj_rec_list=res.get("tail_rec"),
        traj_f1_list=res.get("tail_f1"),
        k=topE,
        reconstruct_step_if_missing=True,
        step_start=1,
        mol=mol,
        native_idx=res.get("native_idx"),
        chain_break_info=chain_break_info,
    )
    inner_csv = f"tailTopE_{topE}_{seed}.csv"
    top_zip = os.path.join(out_dir, f"{inner_csv}.zip")
    write_topE_traj_csv(top_zip, inner_csv, top_records)


def _export_ensemble_for_one_seed(
    out_dir: str,
    seed: int,
    res: Dict[str, Any],
    mol: str,
    chain_break_info: Optional[Tuple[List[int], int]] = None,
):
    seqs = res.get("tail_seq_u8")
    Es   = res.get("tail_E")
    recs = res.get("tail_rec")
    f1s  = res.get("tail_f1")

    if seqs is None or Es is None or len(seqs) == 0 or len(Es) == 0:
        print(f"[WARN] tail ensemble missing for seed={seed}; skip Ensemble CSV ZIP.")
        return

    records = topk_unique_by_min_energy_numpy(
        traj_step_list=None,
        traj_seq_u8_list=seqs,
        traj_E_list=Es,
        traj_rec_list=recs,
        traj_f1_list=f1s,
        k=max(1, len(seqs)),
        reconstruct_step_if_missing=True,
        step_start=1,
        mol=mol,
        native_idx=res.get("native_idx"),
        chain_break_info=chain_break_info,
    )

    inner_csv = f"Ensemble_{seed}.csv"
    out_zip = os.path.join(out_dir, f"{inner_csv}.zip")
    write_ensemble_csv(out_zip, inner_csv, records)


def ensemble_avg_energy_from_tail(
    tail_seq_u8_list,
    tail_E_list,
    beta: float = ENSEMBLE_BETA,
    energy_mode: str = "min",
):
    if tail_seq_u8_list is None or tail_E_list is None:
        return float("nan"), float("nan"), 0
    if len(tail_seq_u8_list) == 0 or len(tail_E_list) == 0:
        return float("nan"), float("nan"), 0
    if len(tail_seq_u8_list) != len(tail_E_list):
        raise ValueError("tail_seq_u8_list and tail_E_list length mismatch")

    beta = float(beta)
    if beta < 0:
        raise ValueError("beta must be >= 0")

    seq_to_Es: Dict[bytes, List[float]] = {}

    for s_u8, E in zip(tail_seq_u8_list, tail_E_list):
        if isinstance(s_u8, torch.Tensor):
            s_u8 = s_u8.detach().cpu().numpy()
        s_u8 = np.asarray(s_u8, dtype=np.uint8)
        key = s_u8.tobytes()
        Ei = float(E)

        if key not in seq_to_Es:
            seq_to_Es[key] = [Ei]
        else:
            seq_to_Es[key].append(Ei)

    if len(seq_to_Es) == 0:
        return float("nan"), float("nan"), 0

    E_rep = []
    for Es in seq_to_Es.values():
        arr = np.asarray(Es, dtype=np.float64)
        if energy_mode == "min":
            E_rep.append(float(np.min(arr)))
        elif energy_mode == "mean":
            E_rep.append(float(np.mean(arr)))
        else:
            raise ValueError(f"unknown energy_mode: {energy_mode}")

    E_rep = np.asarray(E_rep, dtype=np.float64)
    Emin = float(np.min(E_rep))
    shifted = E_rep - Emin
    boltz = np.exp(-beta * shifted)
    Z_tilde = float(np.sum(boltz))

    if Z_tilde <= 0.0 or (not np.isfinite(Z_tilde)):
        return float("nan"), float("nan"), int(len(E_rep))

    probs = boltz / Z_tilde
    E_avg = float(np.sum(E_rep * probs))

    logZ = -beta * Emin + np.log(Z_tilde)
    Z = float(np.exp(logZ)) if logZ < 700 else float("inf")

    return E_avg, Z, int(len(E_rep))


def _drop_big_fields_inplace(res: Dict[str, Any]):
    for k in (
        "traj_step", "traj_rec", "traj_f1", "traj_seq_u8", "traj_E",
        "tail_rec", "tail_f1", "tail_seq_u8", "tail_E",
    ):
        if k in res:
            res[k] = None


def _callable_accepts_param(fn: Callable[..., Any], name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in sig.parameters


def run_single_pdb_once(
    pdb_path: str,
    pot: TriRNASP_Potential,
    device: torch.device,
    steps: int,
    batch_size: int,
    tail_steps: int,
    kl_lambda: float,
    T_min: Optional[float],
    thermo_ss_arg: Optional[str] = None,
    hard_ss_arg: Optional[str] = None,
    ss_penalty: float = 10.0,
    frz_arg: Optional[str] = None,
) -> Dict[str, Any]:
    kwargs = dict(
        pdb_path=pdb_path,
        pot=pot,
        device=device,
        steps=steps,
        batch_size=batch_size,
        tail_steps=tail_steps,
        kl_lambda=kl_lambda,
        top_k=10,
        T_min=T_min,
        thermo_ss_arg=thermo_ss_arg,
        hard_ss_arg=hard_ss_arg,
        ss_penalty=ss_penalty,
    )
    if _callable_accepts_param(design_one_pdb_single, "frz_arg"):
        kwargs["frz_arg"] = frz_arg

    ret = design_one_pdb_single(**kwargs)

    return _parse_dev_single_return(tuple(ret))


def run_single_pdb_multi(
    pdb_path: str,
    seeds: List[int],
    pot: TriRNASP_Potential,
    device: torch.device,
    steps: int,
    batch_size: int,
    tail_steps: int,
    kl_lambda: float,
    T_min: Optional[float],
    thermo_ss_arg: Optional[str] = None,
    hard_ss_arg: Optional[str] = None,
    ss_penalty: float = 10.0,
    frz_arg: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if len(seeds) == 1:
        s = int(seeds[0])
        set_seed(s)
        return [run_single_pdb_once(
            pdb_path=pdb_path,
            pot=pot,
            device=device,
            steps=steps,
            batch_size=batch_size,
            tail_steps=tail_steps,
            kl_lambda=kl_lambda,
            T_min=T_min,
            thermo_ss_arg=thermo_ss_arg,
            hard_ss_arg=hard_ss_arg,
            ss_penalty=ss_penalty,
            frz_arg=frz_arg,
        )]

    if not callable(design_one_pdb_single_multi):
        out = []
        for s in seeds:
            set_seed(int(s))
            out.append(run_single_pdb_once(
                pdb_path=pdb_path,
                pot=pot,
                device=device,
                steps=steps,
                batch_size=batch_size,
                tail_steps=tail_steps,
                kl_lambda=kl_lambda,
                T_min=T_min,
                thermo_ss_arg=thermo_ss_arg,
                hard_ss_arg=hard_ss_arg,
                ss_penalty=ss_penalty,
                frz_arg=frz_arg,
            ))
        return out

    kwargs = dict(
        pdb_path=pdb_path,
        seeds=seeds,
        pot=pot,
        device=device,
        steps=steps,
        batch_size=batch_size,
        tail_steps=tail_steps,
        kl_lambda=kl_lambda,
        top_k=10,
        T_min=T_min,
        thermo_ss_arg=thermo_ss_arg,
        hard_ss_arg=hard_ss_arg,
        ss_penalty=ss_penalty,
    )
    if _callable_accepts_param(design_one_pdb_single_multi, "frz_arg"):
        kwargs["frz_arg"] = frz_arg

    ret = design_one_pdb_single_multi(**kwargs)

    if isinstance(ret, (list, tuple)) and len(ret) == len(seeds) and len(ret) > 0 and isinstance(ret[0], (tuple, list)):
        return [_parse_dev_single_return(tuple(x)) for x in ret]

    if isinstance(ret, dict) and "runs" in ret and isinstance(ret["runs"], (list, tuple)):
        runs = ret["runs"]
        if len(runs) == len(seeds):
            return [_parse_dev_single_return(tuple(x)) for x in runs]

    if isinstance(ret, (list, tuple)) and len(ret) == 2 and isinstance(ret[1], (list, tuple)):
        runs = ret[1]
        if len(runs) == len(seeds):
            return [_parse_dev_single_return(tuple(x)) for x in runs]

    raise RuntimeError("[ERROR] Unrecognized return format from design_one_pdb_single_multi()")


def run_multi_folder_multi(
    multi_dir: str,
    seeds: List[int],
    pot: TriRNASP_Potential,
    device: torch.device,
    steps: int,
    batch_size: int,
    tail_steps: int,
    kl_lambda: float,
    T_min: Optional[float],
    thermo_ss_arg: Optional[str] = None,
    hard_ss_arg: Optional[str] = None,
    ss_penalty: float = 10.0,
    frz_arg: Optional[str] = None,
) -> List[Dict[str, Any]]:
    pdb_paths = glob.glob(os.path.join(multi_dir, "*.pdb"))
    pdb_paths = _sort_multistate_pdbs_target_first(pdb_paths)

    if len(pdb_paths) == 0:
        raise RuntimeError(f"No .pdb files found in folder: {multi_dir}")

    print("[INFO] Multi-state PDB order (target-like first, TM-tagged later):")
    for i, p in enumerate(pdb_paths, 1):
        tag = "TM-tag" if _has_tm_tag_pdb(p) else "target-like"
        print(f"  [{i:02d}] {os.path.basename(p)}   ({tag})")

    if len(pdb_paths) == 1:
        print(f"[WARN] Multi-state folder has only 1 PDB -> treat as single-state: {pdb_paths[0]}")
        return run_single_pdb_multi(
            pdb_path=pdb_paths[0],
            seeds=seeds,
            pot=pot,
            device=device,
            steps=steps,
            batch_size=batch_size,
            tail_steps=tail_steps,
            kl_lambda=kl_lambda,
            T_min=T_min,
            thermo_ss_arg=thermo_ss_arg,
            hard_ss_arg=hard_ss_arg,
            ss_penalty=ss_penalty,
            frz_arg=frz_arg,
        )

    kwargs = dict(
        pdb_paths=pdb_paths,
        seeds=seeds,
        pot=pot,
        device=device,
        steps=steps,
        batch_size=batch_size,
        tail_steps=tail_steps,
        kl_lambda=kl_lambda,
        top_k=10,
        T_min=T_min,
        thermo_ss_arg=thermo_ss_arg,
        hard_ss_arg=hard_ss_arg,
        ss_penalty=ss_penalty,
    )
    if _callable_accepts_param(design_multi_state_multi, "frz_arg"):
        kwargs["frz_arg"] = frz_arg

    ret = design_multi_state_multi(**kwargs)

    if isinstance(ret, (list, tuple)) and len(ret) == len(seeds) and len(ret) > 0 and isinstance(ret[0], (tuple, list)):
        return [_parse_dev_multi_return(tuple(x)) for x in ret]

    if isinstance(ret, dict) and "runs" in ret and isinstance(ret["runs"], (list, tuple)):
        runs = ret["runs"]
        if len(runs) == len(seeds):
            return [_parse_dev_multi_return(tuple(x)) for x in runs]

    if isinstance(ret, (list, tuple)) and len(ret) == 2 and isinstance(ret[1], (list, tuple)):
        runs = ret[1]
        if len(runs) == len(seeds):
            return [_parse_dev_multi_return(tuple(x)) for x in runs]

    raise RuntimeError("[ERROR] Unrecognized return format from design_multi_state_multi()")


def run_streaming_collect_success(
    target_total: int,
    run_fn: Callable[[List[int]], List[Dict[str, Any]]],
    handle_one: Callable[[int, Dict[str, Any]], bool],
    tag: str,
    out_dir: str,
    sleep_s: float = 0.5,
    max_stalled_tries: int = 8,
    max_chunk1_retries: int = 3,
    round_seed_writer: Optional[Callable[[int, List[int]], None]] = None,
) -> int:
    target_total = max(1, int(target_total))
    remaining = target_total
    cur_chunk = remaining
    done_ok = 0

    seeds_since_cleanup = 0
    stalled_tries = 0
    chunk1_retry_count = 0
    round_idx = 0

    while remaining > 0:
        cur_chunk = max(1, min(cur_chunk, remaining))
        seeds = [rand_seed_32bit() for _ in range(cur_chunk)]
        print(f"[INFO] {tag}: running {cur_chunk} seeds (remaining after this: {remaining - cur_chunk})")

        round_ok = 0

        try:
            res_list = run_fn(seeds)
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            if _is_oom_error(e):
                if cur_chunk > 1:
                    old = cur_chunk
                    cur_chunk = max(1, cur_chunk // 2)
                    chunk1_retry_count = 0
                    print(f"[OOM] {tag}: chunk {old} -> {cur_chunk}; hard cleanup then retry")
                    append_fail_log(out_dir, tag, seed=None, stage=f"oom_chunk{old}_split_to_{cur_chunk}", exc=e)
                    _hard_cleanup_sleep(sleep_s=sleep_s)
                else:
                    chunk1_retry_count += 1
                    print(f"[OOM] {tag}: chunk already 1; hard cleanup retry {chunk1_retry_count}/{max_chunk1_retries}")
                    append_fail_log(out_dir, tag, seed=None, stage=f"oom_chunk1_retry_{chunk1_retry_count}", exc=e)
                    _hard_cleanup_sleep(sleep_s=sleep_s)

                    if chunk1_retry_count > max_chunk1_retries:
                        raise RuntimeError(
                            f"[FATAL] {tag}: OOM persists after {chunk1_retry_count} hard-cleanup retries at chunk=1. Stop."
                        ) from e

                stalled_tries += 1
                if stalled_tries > max_stalled_tries:
                    raise RuntimeError(
                        f"[FATAL] {tag}: no progress for {stalled_tries} consecutive rounds "
                        f"(OOM/retry loop). Stop."
                    ) from e

                continue

            append_fail_log(out_dir, tag, seed=None, stage="run_fn_exception", exc=e)

            stalled_tries += 1
            if stalled_tries > max_stalled_tries:
                raise RuntimeError(
                    f"[FATAL] {tag}: no progress for {stalled_tries} consecutive rounds "
                    f"(run_fn exceptions). Stop."
                ) from e

            if sleep_s > 0:
                time.sleep(sleep_s)
            continue

        if not isinstance(res_list, list) or len(res_list) != len(seeds):
            err = RuntimeError(
                f"[ERROR] {tag}: run_fn returned invalid list length: "
                f"got {len(res_list)} expected {len(seeds)}"
            )
            append_fail_log(out_dir, tag, seed=None, stage="bad_return_length", exc=err)

            stalled_tries += 1
            if stalled_tries > max_stalled_tries:
                raise RuntimeError(
                    f"[FATAL] {tag}: no progress for {stalled_tries} consecutive rounds "
                    f"(bad return length). Stop."
                ) from err
            continue

        round_idx += 1
        if round_seed_writer is not None:
            try:
                round_seed_writer(round_idx, seeds)
            except Exception as e:
                print(f"[WARN] Failed to save seed batch for round {round_idx}: {e}")

        for sd, res in zip(seeds, res_list):
            ok = False
            try:
                ok = bool(handle_one(int(sd), res))
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                append_fail_log(out_dir, tag, seed=int(sd), stage="handle_one_exception", exc=e)
                ok = False

            try:
                if isinstance(res, dict):
                    _drop_big_fields_inplace(res)
            except Exception:
                pass
            del res

            if ok:
                done_ok += 1
                remaining -= 1
                round_ok += 1
                if remaining <= 0:
                    break

            seeds_since_cleanup += 1
            _cleanup_cuda()
            if seeds_since_cleanup >= 3:
                seeds_since_cleanup = 0

        del res_list
        del seeds
        _cleanup_cuda()

        if round_ok > 0:
            stalled_tries = 0
            chunk1_retry_count = 0
        else:
            stalled_tries += 1
            print(f"[WARN] {tag}: no successful progress in this round (stalled={stalled_tries}/{max_stalled_tries})")

            if stalled_tries > max_stalled_tries:
                raise RuntimeError(
                    f"[FATAL] {tag}: no successful progress for {stalled_tries} consecutive rounds. Stop."
                )

    return done_ok


def run_exact_one_seed(
    seed: int,
    run_fn: Callable[[List[int]], List[Dict[str, Any]]],
    handle_one: Callable[[int, Dict[str, Any]], bool],
    tag: str,
    out_dir: str,
) -> int:
    seed = int(seed)
    print(f"[INFO] {tag}: exact reproduction mode with seed={seed}")

    try:
        res_list = run_fn([seed])
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        append_fail_log(out_dir, tag, seed=seed, stage="exact_run_fn_exception", exc=e)
        raise

    if not isinstance(res_list, list) or len(res_list) != 1:
        err = RuntimeError(
            f"[ERROR] {tag}: exact one-seed mode expects one result, got len={len(res_list) if isinstance(res_list, list) else 'non-list'}"
        )
        append_fail_log(out_dir, tag, seed=seed, stage="exact_bad_return_length", exc=err)
        raise err

    res = res_list[0]
    try:
        ok = bool(handle_one(seed, res))
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        append_fail_log(out_dir, tag, seed=seed, stage="exact_handle_one_exception", exc=e)
        raise
    finally:
        try:
            if isinstance(res, dict):
                _drop_big_fields_inplace(res)
        except Exception:
            pass
        _cleanup_cuda()

    return 1 if ok else 0


def run_exact_seed_batch(
    seeds: List[int],
    run_fn: Callable[[List[int]], List[Dict[str, Any]]],
    handle_one: Callable[[int, Dict[str, Any]], bool],
    tag: str,
    out_dir: str,
) -> int:
    if seeds is None or len(seeds) == 0:
        raise ValueError("[ERROR] run_exact_seed_batch requires a non-empty seed list")

    seeds = [int(s) for s in seeds]
    print(f"[INFO] {tag}: exact seed-batch replay mode with {len(seeds)} seeds")

    try:
        res_list = run_fn(seeds)
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        append_fail_log(out_dir, tag, seed=None, stage="exact_seed_batch_run_fn_exception", exc=e)
        raise

    if not isinstance(res_list, list) or len(res_list) != len(seeds):
        err = RuntimeError(
            f"[ERROR] {tag}: exact seed-batch mode expects {len(seeds)} results, got "
            f"{len(res_list) if isinstance(res_list, list) else 'non-list'}"
        )
        append_fail_log(out_dir, tag, seed=None, stage="exact_seed_batch_bad_return_length", exc=err)
        raise err

    ok_total = 0
    for sd, res in zip(seeds, res_list):
        try:
            ok = bool(handle_one(int(sd), res))
            if ok:
                ok_total += 1
        except KeyboardInterrupt:
            raise
        except BaseException as e:
            append_fail_log(out_dir, tag, seed=int(sd), stage="exact_seed_batch_handle_one_exception", exc=e)
            raise
        finally:
            try:
                if isinstance(res, dict):
                    _drop_big_fields_inplace(res)
            except Exception:
                pass
            _cleanup_cuda()

    return ok_total


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "DS3dRNA: de novo sequence design and fixed-backbone sequence "
            "ranking for nucleic acids."
        ),
    )

    p.add_argument("-m", "--multi", action="store_true",
                   help="Design mode: treat INPUT as a multi-state folder.")
    p.add_argument("input_path", nargs="?", default=None,
                   help="Design input: path to a PDB file OR a directory of PDBs OR (with -m) a multi-state folder.")

    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=320)
    p.add_argument("--tail_steps", type=int, default=4000)
    p.add_argument("--kl_lambda", type=float, default=0.0)

    p.add_argument("--temp", type=float, default=DEFAULT_T_MIN,
                   help="Simulated annealing minimum temperature T_min (default/final fixed value: 4.5).")

    p.add_argument(
        "--mol",
        type=str,
        default="RNA",
        choices=["RNA", "DNA"],
        help="Molecule mode: RNA (default) or DNA. Controls Energy/RNA vs Energy/DNA and output U/T display.",
    )

    p.add_argument("--batch", type=int, default=10,
                   help="Total number of SUCCESSFUL runs (seeds) requested per target.")
    p.add_argument("--seed", type=int, default=None,
                   help="Exact reproduction mode: run ONLY this one seed. When specified, --batch is ignored.")
    p.add_argument("--seed_batch", type=str, default=None,
                   help="Replay an exact ordered seed batch from file. When specified, --batch is ignored.")
    p.add_argument("--topE", type=int, default=10000,
                   help="trajTopE: keep top-K unique seq by min E over WHOLE run.")

    p.add_argument("--debug_traj", action="store_true", help="Print traj/tail types and lengths for debugging.")

    p.add_argument("-rank", "--rank", action="store_true", help="Rank mode (Fine energy only).")
    p.add_argument("--str", type=str, default=None, help="Rank mode: structure path (PDB or dir).")
    p.add_argument("--fa", type=str, default=None, help="Rank mode: FASTA file.")
    p.add_argument("--rank_out", type=str, default=None, help="Rank mode: output CSV path/dir (optional).")
    p.add_argument("--batch_rank", action="store_true", help="Rank mode: --str is a directory; one CSV per PDB.")
    p.add_argument("--multi_rank", action="store_true", help="Rank mode: --str is a multi-state directory; one CSV.")
    p.add_argument("--rank_bs", type=int, default=10240, help="Rank mode batch size (default: 10240).")

    p.add_argument(
        "--ss",
        type=str,
        default=None,
        help=(
            "Secondary-structure input source. "
            "Design: omitted means auto-parsed contacts for thermodynamic scoring only; "
            "'auto' applies parsed contacts to thermodynamic and hard constraints; "
            "'none' supplies an all-unpaired DBN (zero contacts). "
            "Rank: omitted disables SS; 'auto' parses contacts; 'none' means zero base pairs. "
            "A dot-bracket string, .dbn file, or matched .dbn directory is also accepted."
        ),
    )
    p.add_argument(
        "--frz",
        type=str,
        default=None,
        help=(
            "Frozen sequence mask for design mode. "
            "A/U/C/G/T positions are locked; '-' positions are free; '&' is ignored as a chain separator. "
            "Can be a direct mask string, a file, or in batch/multi use a directory containing matched "
            ".frz/.frozen/.txt/.fa/.fasta files."
        ),
    )
    p.add_argument(
        "--ss_penalty",
        type=float,
        default=10000.0,
        help="Penalty added per violated base pair in the secondary-structure constraint.",
    )

    return p.parse_args()


def _resolve_rank_out_default_single(pdb_path: str, fasta_path: str, rank_out: Optional[str]) -> str:
    if callable(_default_rank_out_for_pdb):
        try:
            return _default_rank_out_for_pdb(pdb_path, fasta_path, rank_out)
        except TypeError:
            pass

    base = os.path.splitext(os.path.basename(pdb_path))[0]
    out_dir = os.path.dirname(os.path.abspath(fasta_path)) or "."
    if rank_out:
        ro = os.path.abspath(os.path.expanduser(rank_out))
        if ro.endswith(os.sep) or os.path.isdir(ro) or (not ro.lower().endswith(".csv")):
            ensure_dir(ro)
            return os.path.join(ro, f"{base}.rank.csv")
        return ro
    return os.path.join(out_dir, f"{base}.rank.csv")


def _resolve_rank_out_default_multi(multi_dir: str, rank_out: Optional[str]) -> str:
    multi_dir = os.path.abspath(os.path.expanduser(multi_dir))
    base = os.path.basename(os.path.abspath(multi_dir.rstrip("/\\")))
    if rank_out:
        ro = os.path.abspath(os.path.expanduser(rank_out))
        if ro.endswith(os.sep) or os.path.isdir(ro) or (not ro.lower().endswith(".csv")):
            ensure_dir(ro)
            return os.path.join(ro, f"{base}-Multi.rank.csv")
        return ro
    return os.path.join(multi_dir, f"{base}-Multi.rank.csv")


def _resolve_ss_for_single(ss_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if _is_none_ss_arg(ss_arg):
        return no_contact_ss_for_pdb(pdb_path)
    if _use_auto_ss(ss_arg):
        return auto_parse_ss_for_pdb(pdb_path)
    return ss_arg


def _resolve_ss_for_batch(ss_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if _is_none_ss_arg(ss_arg):
        return no_contact_ss_for_pdb(pdb_path)
    if _use_auto_ss(ss_arg):
        return auto_parse_ss_for_pdb(pdb_path)

    if looks_like_dotbracket(ss_arg):
        return ss_arg

    if os.path.isfile(ss_arg):
        return ss_arg

    if os.path.isdir(ss_arg):
        base = os.path.splitext(os.path.basename(pdb_path))[0]
        cands = [
            os.path.join(ss_arg, base + ".dbn"),
            os.path.join(ss_arg, base + ".ss"),
            os.path.join(ss_arg, base + ".txt"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched SS file for {pdb_path} under directory: {ss_arg}")

    raise FileNotFoundError(f"Invalid --ss for batch mode: {ss_arg}")


def _resolve_ss_for_multi(ss_arg: Optional[str], multi_dir: str, ref_pdb_path: Optional[str] = None) -> Optional[str]:
    if _is_none_ss_arg(ss_arg):
        if not ref_pdb_path:
            raise FileNotFoundError(f"Cannot build zero-contact SS for multi-dir {multi_dir}: no PDBs found.")
        return no_contact_ss_for_pdb(ref_pdb_path)
    if _use_auto_ss(ss_arg):
        if not ref_pdb_path:
            raise FileNotFoundError(f"Cannot auto-parse SS for multi-dir {multi_dir}: no PDBs found.")
        # We parse the secondary structure based on the target-like structure (the first PDB)
        return auto_parse_ss_for_pdb(ref_pdb_path)

    if looks_like_dotbracket(ss_arg):
        return ss_arg

    if os.path.isfile(ss_arg):
        return ss_arg

    if os.path.isdir(ss_arg):
        folder = os.path.basename(os.path.normpath(multi_dir))
        cands = [
            os.path.join(ss_arg, folder + ".dbn"),
            os.path.join(ss_arg, folder + ".ss"),
            os.path.join(ss_arg, folder + ".txt"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched multi-state SS file for folder {multi_dir} under: {ss_arg}")

    raise FileNotFoundError(f"Invalid --ss for multi-state mode: {ss_arg}")


def _resolve_design_ss_dual_single(ss_arg: Optional[str], pdb_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Design mode:
      - ss_arg is None  -> implicit auto only for thermo; hard penalty disabled
      - explicit --ss   -> same resolved SS goes to both thermo and hard penalty
    """
    if ss_arg is None:
        thermo_ss_arg = auto_parse_ss_for_pdb(pdb_path)
        hard_ss_arg = None
        return thermo_ss_arg, hard_ss_arg

    if _is_auto_ss_arg(ss_arg):
        resolved = auto_parse_ss_for_pdb(pdb_path)
        return resolved, resolved

    if _is_none_ss_arg(ss_arg):
        resolved = no_contact_ss_for_pdb(pdb_path)
        return resolved, resolved

    resolved = _resolve_ss_for_single(ss_arg, pdb_path)
    return resolved, resolved


def _resolve_design_ss_dual_batch(ss_arg: Optional[str], pdb_path: str) -> Tuple[Optional[str], Optional[str]]:
    if ss_arg is None:
        thermo_ss_arg = auto_parse_ss_for_pdb(pdb_path)
        hard_ss_arg = None
        return thermo_ss_arg, hard_ss_arg

    if _is_auto_ss_arg(ss_arg):
        resolved = auto_parse_ss_for_pdb(pdb_path)
        return resolved, resolved

    if _is_none_ss_arg(ss_arg):
        resolved = no_contact_ss_for_pdb(pdb_path)
        return resolved, resolved

    resolved = _resolve_ss_for_batch(ss_arg, pdb_path)
    return resolved, resolved


def _resolve_design_ss_dual_multi(
    ss_arg: Optional[str],
    multi_dir: str,
    ref_pdb_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if ss_arg is None:
        if not ref_pdb_path:
            raise FileNotFoundError(f"Cannot auto-parse thermo SS for multi-dir {multi_dir}: no PDBs found.")
        thermo_ss_arg = auto_parse_ss_for_pdb(ref_pdb_path)
        hard_ss_arg = None
        return thermo_ss_arg, hard_ss_arg

    if _is_auto_ss_arg(ss_arg):
        if not ref_pdb_path:
            raise FileNotFoundError(f"Cannot auto-parse SS for multi-dir {multi_dir}: no PDBs found.")
        resolved = auto_parse_ss_for_pdb(ref_pdb_path)
        return resolved, resolved

    if _is_none_ss_arg(ss_arg):
        if not ref_pdb_path:
            raise FileNotFoundError(f"Cannot build zero-contact SS for multi-dir {multi_dir}: no PDBs found.")
        resolved = no_contact_ss_for_pdb(ref_pdb_path)
        return resolved, resolved

    resolved = _resolve_ss_for_multi(ss_arg, multi_dir, ref_pdb_path)
    return resolved, resolved


def _looks_like_frz_mask(s: str) -> bool:
    if s is None:
        return False
    allowed = set("AUCGTaucgt-._& \t\r\n")
    return all(ch in allowed for ch in str(s))


def _resolve_frz_for_single(frz_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if frz_arg is None:
        return None
    return frz_arg


def _resolve_frz_for_batch(frz_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if frz_arg is None:
        return None

    if _looks_like_frz_mask(frz_arg):
        return frz_arg

    if os.path.isfile(frz_arg):
        return frz_arg

    if os.path.isdir(frz_arg):
        base = os.path.splitext(os.path.basename(pdb_path))[0]
        cands = [
            os.path.join(frz_arg, base + ".frz"),
            os.path.join(frz_arg, base + ".frozen"),
            os.path.join(frz_arg, base + ".txt"),
            os.path.join(frz_arg, base + ".fa"),
            os.path.join(frz_arg, base + ".fasta"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched FRZ file for {pdb_path} under directory: {frz_arg}")

    raise FileNotFoundError(f"Invalid --frz for batch mode: {frz_arg}")


def _resolve_frz_for_multi(frz_arg: Optional[str], multi_dir: str) -> Optional[str]:
    if frz_arg is None:
        return None

    if _looks_like_frz_mask(frz_arg):
        return frz_arg

    if os.path.isfile(frz_arg):
        return frz_arg

    if os.path.isdir(frz_arg):
        folder = os.path.basename(os.path.normpath(multi_dir))
        cands = [
            os.path.join(frz_arg, folder + ".frz"),
            os.path.join(frz_arg, folder + ".frozen"),
            os.path.join(frz_arg, folder + ".txt"),
            os.path.join(frz_arg, folder + ".fa"),
            os.path.join(frz_arg, folder + ".fasta"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched multi-state FRZ file for folder {multi_dir} under: {frz_arg}")

    raise FileNotFoundError(f"Invalid --frz for multi-state mode: {frz_arg}")


def _resolve_ss_for_single_rank(ss_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if ss_arg is None:
        return None
    if _is_none_ss_arg(ss_arg):
        return no_contact_ss_for_pdb(pdb_path)
    if _is_auto_ss_arg(ss_arg):
        return auto_parse_ss_for_pdb(pdb_path)
    return ss_arg


def _resolve_ss_for_batch_rank(ss_arg: Optional[str], pdb_path: str) -> Optional[str]:
    if ss_arg is None:
        return None
    if _is_none_ss_arg(ss_arg):
        return no_contact_ss_for_pdb(pdb_path)
    if _is_auto_ss_arg(ss_arg):
        return auto_parse_ss_for_pdb(pdb_path)

    if looks_like_dotbracket(ss_arg):
        return ss_arg

    if os.path.isfile(ss_arg):
        return ss_arg

    if os.path.isdir(ss_arg):
        base = os.path.splitext(os.path.basename(pdb_path))[0]
        cands = [
            os.path.join(ss_arg, base + ".dbn"),
            os.path.join(ss_arg, base + ".ss"),
            os.path.join(ss_arg, base + ".txt"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched SS file for {pdb_path} under directory: {ss_arg}")

    raise FileNotFoundError(f"Invalid --ss for batch rank mode: {ss_arg}")


def _resolve_ss_for_multi_rank(ss_arg: Optional[str], multi_dir: str, ref_pdb_path: Optional[str] = None) -> Optional[str]:
    if ss_arg is None:
        return None
    if _is_none_ss_arg(ss_arg):
        if not ref_pdb_path:
            raise FileNotFoundError(f"Cannot build zero-contact SS for multi-dir {multi_dir}: no PDBs found.")
        return no_contact_ss_for_pdb(ref_pdb_path)
    if _is_auto_ss_arg(ss_arg):
        if not ref_pdb_path:
            raise FileNotFoundError(f"Cannot auto-parse SS for multi-dir {multi_dir}: no PDBs found.")
        return auto_parse_ss_for_pdb(ref_pdb_path)

    if looks_like_dotbracket(ss_arg):
        return ss_arg

    if os.path.isfile(ss_arg):
        return ss_arg

    if os.path.isdir(ss_arg):
        folder = os.path.basename(os.path.normpath(multi_dir))
        cands = [
            os.path.join(ss_arg, folder + ".dbn"),
            os.path.join(ss_arg, folder + ".ss"),
            os.path.join(ss_arg, folder + ".txt"),
        ]
        for p in cands:
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"No matched multi-state SS file for folder {multi_dir} under: {ss_arg}")

    raise FileNotFoundError(f"Invalid --ss for multi rank mode: {ss_arg}")


def _extract_dotbracket_from_text(text: str) -> Optional[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">") or line.startswith("#") or line.startswith(";"):
            continue
        if looks_like_dotbracket(line):
            return line
    return None


def _is_auto_ss_arg(ss_arg: Optional[str]) -> bool:
    return bool(ss_arg is not None and str(ss_arg).strip().lower() == "auto")


def _is_none_ss_arg(ss_arg: Optional[str]) -> bool:
    return bool(ss_arg is not None and str(ss_arg).strip().lower() == "none")


def _use_auto_ss(ss_arg: Optional[str]) -> bool:
    return (ss_arg is None) or _is_auto_ss_arg(ss_arg)


def _resolve_ss_preview(ss_arg: Optional[str]) -> Tuple[str, Optional[str], Optional[str]]:
    if ss_arg is None:
        return ("disabled", None, None)

    if _is_auto_ss_arg(ss_arg):
        return ("auto", "<on-the-fly via coarse_grained_ss_parser>", None)

    if _is_none_ss_arg(ss_arg):
        return ("none", "<generated all-unpaired DBN>", None)

    if looks_like_dotbracket(ss_arg):
        dbn = ss_arg.strip()
        return ("inline", "<inline>", dbn)

    path = os.path.abspath(os.path.expanduser(ss_arg))
    if os.path.isfile(path):
        dbn = None
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                dbn = _extract_dotbracket_from_text(f.read())
        except Exception:
            dbn = None
        return ("file", path, dbn)
    if os.path.isdir(path):
        return ("directory", path, None)

    return ("unknown", path, None)


def _read_frz_preview_text(frz_arg: str) -> str:
    frz_arg = str(frz_arg).strip()
    if os.path.isfile(frz_arg):
        parts = []
        with open(frz_arg, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">") or line.startswith("#"):
                    continue
                parts.append(line)
        return "".join(parts)
    return frz_arg


def _frz_preview_compact(frz_arg: Optional[str]) -> Tuple[str, int, int]:
    if frz_arg is None:
        return "", 0, 0

    raw = _read_frz_preview_text(frz_arg)
    chars = []
    n_fixed = 0

    for ch in raw:
        if ch.isspace() or ch == "&":
            continue
        if ch in "-._":
            chars.append("-")
        elif ch in "AUCGTaucgt":
            b = ch.upper().replace("T", "U")
            chars.append(b)
            n_fixed += 1
        else:
            chars.append("?")

    compact = "".join(chars)
    return compact, n_fixed, len(compact)


def _print_frz_preview(scope: str, frz_arg: Optional[str]):
    if frz_arg is None:
        print(f"[INFO][frz] {scope}: disabled")
        return

    if _looks_like_frz_mask(frz_arg):
        source_kind = "inline"
        source_path = "<inline>"
    else:
        source_path = os.path.abspath(os.path.expanduser(frz_arg))
        if os.path.isfile(source_path):
            source_kind = "file"
        elif os.path.isdir(source_path):
            source_kind = "directory"
        else:
            source_kind = "unknown"

    print(f"[INFO][frz] {scope}: enabled")
    print(f"[INFO][frz] {scope}: source = {source_kind}")
    print(f"[INFO][frz] {scope}: path = {source_path}")

    if source_kind != "directory":
        compact, n_fixed, compact_len = _frz_preview_compact(frz_arg)
        print(f"[INFO][frz] {scope}: n_fixed = {n_fixed}/{compact_len}")
        print(f"[INFO][frz] {scope}: compact mask = {compact}")


def _print_ss_preview(
    scope: str,
    ss_arg: Optional[str],
    *,
    pdb_path: Optional[str] = None,
    original_ss_arg: Optional[str] = None,
):
    abs_pdb = os.path.abspath(os.path.expanduser(pdb_path)) if pdb_path else None
    meta = _AUTO_SS_CACHE.get(abs_pdb) if abs_pdb else None

    if _is_none_ss_arg(original_ss_arg):
        print(f"[INFO][ss] {scope}: enabled")
        print(f"[INFO][ss] {scope}: source = none")
        if pdb_path is not None:
            print(f"[INFO][ss] {scope}: pdb = {os.path.basename(abs_pdb)}")
        print(f"[INFO][ss] {scope}: pairs = 0")
        print(f"[INFO][ss] {scope}: dotbracket = {ss_arg or '<unavailable>'}")
        return

    # Auto / implicit-auto branch.
    if (original_ss_arg is None) or _is_auto_ss_arg(original_ss_arg):
        source_label = "auto-implicit" if original_ss_arg is None else "auto"

        print(f"[INFO][ss] {scope}: enabled")
        print(f"[INFO][ss] {scope}: source = {source_label}")
        print(f"[INFO][ss] {scope}: parser = coarse_grained_ss_parser")

        if meta is not None:
            preset_name = meta.get("preset_name", "conservative")
            print(f"[INFO][ss] {scope}: preset = {preset_name}")
            print(f"[INFO][ss] {scope}: pdb = {meta.get('pdb_name', '<unknown>')}")
            print(f"[INFO][ss] {scope}: nts = {meta.get('nts', '<unknown>')}")

            p = meta.get("params", AUTO_SS_PRESET)
            print(
                f"[INFO][ss] {scope}: params = "
                f"WC[{float(p.get('wc_nn_min', 8.0)):.1f},{float(p.get('wc_nn_max', 9.5)):.1f}] "
                f"ideal={float(p.get('wc_nn_ideal', 8.84)):.2f} ; "
                f"nonWC[{float(p.get('nonwc_nn_min', 7.5)):.1f},{float(p.get('nonwc_nn_max', 11.0)):.1f}] ; "
                f"gap={float(p.get('min_score_gap', 0.20)):.2f} ; "
                f"GU={'on' if bool(p.get('allow_gu_wobble', True)) else 'off'} ; "
                f"PK={'on' if bool(p.get('allow_pseudoknot', True)) else 'off'}"
            )

            auto_mode = meta.get("auto_mode", "unknown_seq" if HAS_UNKNOWN_SEQ_SS else "native_seq_fallback")
            if auto_mode == "unknown_seq":
                print(
                    f"[INFO][ss] {scope}: auto_seq = "
                    f"unknown-N ; pairs={meta.get('pair_count', '<na>')}"
                )
            else:
                print(
                    f"[INFO][ss] {scope}: auto_seq = "
                    f"native-seq-fallback ; pairs={meta.get('pair_count', '<na>')}"
                )

            print(f"[INFO][ss] {scope}: dotbracket = {meta.get('dbn', '<unavailable>')}")
            return

        # Auto branch but cache unavailable. This should be rare, but keep it robust.
        preset_name = AUTO_SS_MODE_NAME if globals().get("HAS_UNKNOWN_SEQ_SS", False) else "conservative-native-seq-fallback"
        print(f"[INFO][ss] {scope}: preset = {preset_name}")

        if pdb_path is not None:
            print(f"[INFO][ss] {scope}: pdb = {os.path.basename(abs_pdb)}")

        p = AUTO_SS_PRESET
        print(
            f"[INFO][ss] {scope}: params = "
            f"WC[{float(p.get('wc_nn_min', 8.0)):.1f},{float(p.get('wc_nn_max', 9.5)):.1f}] "
            f"ideal={float(p.get('wc_nn_ideal', 8.84)):.2f} ; "
            f"nonWC[{float(p.get('nonwc_nn_min', 7.5)):.1f},{float(p.get('nonwc_nn_max', 11.0)):.1f}] ; "
            f"gap={float(p.get('min_score_gap', 0.20)):.2f} ; "
            f"GU={'on' if bool(p.get('allow_gu_wobble', True)) else 'off'} ; "
            f"PK={'on' if bool(p.get('allow_pseudoknot', True)) else 'off'}"
        )

        if globals().get("HAS_UNKNOWN_SEQ_SS", False):
            print(f"[INFO][ss] {scope}: auto_seq = unknown-N ; pairs=<unavailable>")
        else:
            print(f"[INFO][ss] {scope}: auto_seq = native-seq-fallback ; pairs=<unavailable>")

        if ss_arg is not None:
            print(f"[INFO][ss] {scope}: dotbracket = {ss_arg}")
        else:
            print(f"[INFO][ss] {scope}: dotbracket = <unavailable>")
        return

    # Explicit user-provided --ss branch.
    source_kind, dbn_path, dbn = _resolve_ss_preview(ss_arg)

    print(f"[INFO][ss] {scope}: enabled")
    print(f"[INFO][ss] {scope}: source = {source_kind}")

    if dbn_path is not None:
        print(f"[INFO][ss] {scope}: dbn_path = {dbn_path}")
    else:
        print(f"[INFO][ss] {scope}: dbn_path = <none>")

    if dbn is not None:
        print(f"[INFO][ss] {scope}: dotbracket = {dbn}")
    else:
        print(f"[INFO][ss] {scope}: dotbracket = <unavailable>")


def _print_design_ss_preview(
    scope_prefix: str,
    thermo_ss_arg: Optional[str],
    hard_ss_arg: Optional[str],
    *,
    pdb_path: Optional[str] = None,
    original_ss_arg: Optional[str] = None,
):
    """
    Pretty preview for design-mode dual SS channels.

    Cases
    -----
    1) thermo == None and hard == None
       -> print disabled once
    2) thermo == hard (common explicit --ss case)
       -> print once, used_for = thermo + hard
    3) thermo != hard
       -> print thermo and hard separately
    """
    if thermo_ss_arg is None and hard_ss_arg is None:
        print(f"[INFO][ss] {scope_prefix}: disabled")
        return

    # same object / same resolved DBN / same path string
    if thermo_ss_arg == hard_ss_arg and thermo_ss_arg is not None:
        print(f"[INFO][ss] {scope_prefix}: used_for = thermo + hard")
        _print_ss_preview(
            scope_prefix,
            thermo_ss_arg,
            pdb_path=pdb_path,
            original_ss_arg=original_ss_arg,
        )
        return

    # split display
    if thermo_ss_arg is None:
        print(f"[INFO][ss] {scope_prefix}-thermo: disabled")
    else:
        _print_ss_preview(
            f"{scope_prefix}-thermo",
            thermo_ss_arg,
            pdb_path=pdb_path,
            original_ss_arg=original_ss_arg,
        )

    if hard_ss_arg is None:
        print(f"[INFO][ss] {scope_prefix}-hard: disabled")
    else:
        _print_ss_preview(
            f"{scope_prefix}-hard",
            hard_ss_arg,
            pdb_path=pdb_path,
            original_ss_arg=original_ss_arg,
        )

def main():
    global dev, DEV_PATH
    global print_trirnade_banner
    global set_seed, design_one_pdb_single, write_top10_energy_csv
    global run_rank_mode, run_rank_batch, run_rank_multi
    global _default_rank_out_for_pdb
    global design_one_pdb_single_multi, design_multi_state_multi

    args = parse_args()
    mol_mode = normalize_mol(args.mol)

    dev, DEV_PATH = load_dev_core(mol_mode)

    required = [
        "set_seed",
        "design_one_pdb_single",
        "print_trirnade_banner",
        "write_top10_energy_csv",
        "run_rank_mode",
        "run_rank_batch",
        "run_rank_multi",
        "design_one_pdb_single_multi",
        "design_multi_state_multi",
    ]
    missing = [x for x in required if not hasattr(dev, x)]
    if missing:
        raise RuntimeError(f"Invalid computational core: missing symbols: {missing}")

    print_trirnade_banner = dev.print_trirnade_banner

    set_seed = dev.set_seed
    design_one_pdb_single = dev.design_one_pdb_single
    write_top10_energy_csv = dev.write_top10_energy_csv

    run_rank_mode = dev.run_rank_mode
    run_rank_batch = dev.run_rank_batch
    run_rank_multi = dev.run_rank_multi
    _default_rank_out_for_pdb = getattr(dev, "_default_rank_out_for_pdb", None)

    design_one_pdb_single_multi = getattr(dev, "design_one_pdb_single_multi", None)
    design_multi_state_multi = getattr(dev, "design_multi_state_multi", None)

    if getattr(args, "frz", None) is not None:
        for fn_name, fn in (
            ("design_one_pdb_single", design_one_pdb_single),
            ("design_one_pdb_single_multi", design_one_pdb_single_multi),
            ("design_multi_state_multi", design_multi_state_multi),
        ):
            if callable(fn) and not _callable_accepts_param(fn, "frz_arg"):
                raise RuntimeError(
                    f"[ERROR] Loaded computational core does not support --frz: "
                    f"{fn_name}() has no frz_arg parameter. Core path: {DEV_PATH}"
                )

    print_trirnade_banner()

    if args.seed is not None and args.seed_batch is not None:
        raise RuntimeError("[ERROR] --seed and --seed_batch are mutually exclusive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Molecule mode: {mol_mode}")

    energy_root = os.path.join("Energy", mol_mode)
    suffix = "_D" if mol_mode == "DNA" else ""
    rough_path = resolve_energy_file(energy_root, f"Rough{suffix}")
    fine_path = resolve_energy_file(energy_root, f"Fine{suffix}")

    print(f"[INFO] Rough energy: {rough_path}")
    print(f"[INFO] Fine  energy: {fine_path}")

    pot = TriRNASP_Potential(
        rough_path=rough_path,
        fine_path=fine_path,
        D=12,
        R0=9.500,
        bw_rough=1.35714,
        bw_fine=0.52778,
        device=device,
    )

    ss_arg_cli: Optional[str] = args.ss
    frz_arg_cli: Optional[str] = args.frz
    ss_penalty: float = float(args.ss_penalty)
    ss_mode_explicit = ss_arg_cli is not None

    T_min: Optional[float] = float(args.temp) if args.temp is not None else None
    reproduce_seed: Optional[int] = None if args.seed is None else int(args.seed)
    reproduce_seed_batch: Optional[List[int]] = None
    if args.seed_batch is not None:
        reproduce_seed_batch = read_seed_batch_file(args.seed_batch)

    batch_size = int(args.batch_size)
    kl_lambda = float(args.kl_lambda)
    rank_bs = int(args.rank_bs)

    replay_mode = (reproduce_seed is not None) or (reproduce_seed_batch is not None)
    repeat_runs_total = max(1, int(args.batch))

    print(f"[INFO] Final fixed T_min         : {T_min}")
    print(f"[INFO] Final fixed batch_size    : {batch_size}")

    if reproduce_seed is not None:
        print(f"[INFO] Exact reproduction mode enabled: seed={reproduce_seed}")
        print("[INFO] --batch is ignored because --seed is specified.")
    elif reproduce_seed_batch is not None:
        print(f"[INFO] Exact seed-batch replay mode enabled: {len(reproduce_seed_batch)} seeds")
        print("[INFO] --batch is ignored because --seed_batch is specified.")
    else:
        print(f"[INFO] Requested SUCCESSFUL runs per target (--batch): {repeat_runs_total}")

    if args.rank:
        if args.str is None or args.fa is None:
            raise RuntimeError("[ERROR] -rank requires --str <pdb_or_dir> and --fa <fasta>")
        if args.batch_rank and args.multi_rank:
            raise RuntimeError("[ERROR] --batch_rank and --multi_rank are mutually exclusive")
        if args.seed is not None or args.seed_batch is not None:
            print("[WARN] --seed / --seed_batch are ignored in rank mode.")
        if frz_arg_cli is not None:
            print("[INFO][frz] rank mode: --frz is ignored because ranking does not mutate or decode sequences.")
        print("[INFO] Rank mode: no thermodynamic reranker; --ss only provides hard SS constraint filtering.")

        str_path = os.path.abspath(os.path.expanduser(args.str.strip()))
        fa_path = os.path.abspath(os.path.expanduser(args.fa.strip()))
        if not os.path.isfile(fa_path):
            raise FileNotFoundError(f"[ERROR] --fa not found: {fa_path}")

        if args.batch_rank:
            if not os.path.isdir(str_path):
                raise RuntimeError(f"[ERROR] --batch_rank expects --str to be a directory, got: {str_path}")

            pdb_list = sorted(glob.glob(os.path.join(str_path, "*.pdb")))
            if not pdb_list:
                raise RuntimeError(f"[ERROR] No .pdb files found under: {str_path}")

            for pdb_path in pdb_list:
                ss_arg_this = _resolve_ss_for_batch_rank(ss_arg_cli, pdb_path)
                if ss_arg_this is None:
                    print(f"[INFO][ss] rank-batch:{os.path.basename(pdb_path)}: disabled")
                else:
                    _print_ss_preview(
                        f"rank-batch:{os.path.basename(pdb_path)}",
                        ss_arg_this,
                        pdb_path=pdb_path,
                        original_ss_arg=ss_arg_cli,
                    )
                out_csv = _resolve_rank_out_default_single(pdb_path, fa_path, args.rank_out)
                chain_break_info = chain_break_info_from_ss(ss_arg_this)
                run_rank_mode(
                    pdb_path=pdb_path,
                    fasta_path=fa_path,
                    pot=pot,
                    device=device,
                    out_csv=out_csv,
                    rank_bs=rank_bs,
                    ss_arg=ss_arg_this,
                    ss_penalty=ss_penalty,
                )
                postprocess_rank_csv_sequences(out_csv, mol_mode, chain_break_info)
            return

        if args.multi_rank:
            pdb_paths = glob.glob(os.path.join(str_path, "*.pdb"))
            pdb_paths = _sort_multistate_pdbs_target_first(pdb_paths)
            ref_pdb = pdb_paths[0] if pdb_paths else None

            ss_arg_this = _resolve_ss_for_multi_rank(ss_arg_cli, str_path, ref_pdb)
            if ss_arg_this is None:
                print(f"[INFO][ss] rank-multi:{os.path.basename(os.path.normpath(str_path))}: disabled")
            else:
                _print_ss_preview(
                    f"rank-multi:{os.path.basename(os.path.normpath(str_path))}",
                    ss_arg_this,
                    pdb_path=ref_pdb,
                    original_ss_arg=ss_arg_cli,
                )
            out_csv = _resolve_rank_out_default_multi(str_path, args.rank_out)
            chain_break_info = chain_break_info_from_ss(ss_arg_this)
            run_rank_multi(
                multi_dir=str_path,
                fasta_path=fa_path,
                pot=pot,
                device=device,
                rank_out=args.rank_out,
                rank_bs=rank_bs,
                ss_arg=ss_arg_this,
                ss_penalty=ss_penalty,
            )
            postprocess_rank_csv_sequences(out_csv, mol_mode, chain_break_info)
            return

        ss_arg_this = _resolve_ss_for_single_rank(ss_arg_cli, str_path)
        if ss_arg_this is None:
            print(f"[INFO][ss] rank-single:{os.path.basename(str_path)}: disabled")
        else:
            _print_ss_preview(
                f"rank-single:{os.path.basename(str_path)}",
                ss_arg_this,
                pdb_path=str_path,
            original_ss_arg=ss_arg_cli,
        )
        out_csv = _resolve_rank_out_default_single(str_path, fa_path, args.rank_out)
        chain_break_info = chain_break_info_from_ss(ss_arg_this)
        run_rank_mode(
            pdb_path=str_path,
            fasta_path=fa_path,
            pot=pot,
            device=device,
            out_csv=out_csv,
            rank_bs=rank_bs,
            ss_arg=ss_arg_this,
            ss_penalty=ss_penalty,
        )
        postprocess_rank_csv_sequences(out_csv, mol_mode, chain_break_info)
        return

    if ss_mode_explicit:
        steps = SS_STEPS
        tail_steps = SS_TAIL_STEPS
        topE = SS_TOPE
        print(f"[INFO][ss] explicit --ss detected: large-trajectory override active: steps={SS_STEPS}, tail_steps={SS_TAIL_STEPS}, topE={SS_TOPE}")
    else:
        steps = int(args.steps)
        tail_steps = int(args.tail_steps)
        topE = max(1, int(args.topE))
        print("[INFO][ss] default implicit auto-SS enabled; user steps/tail_steps/topE are preserved.")

    print(f"[INFO] trajTopE (trajectory) = {topE}")

    if args.input_path is None:
        raise RuntimeError("[ERROR] No input_path provided. For rank, use -rank; for design, provide a PDB/dir.")

    input_path = os.path.abspath(os.path.expanduser(args.input_path))

    if args.multi:
        if not os.path.isdir(input_path):
            raise RuntimeError(f"[ERROR] -m expects a directory, got: {input_path}")

        folder_name = os.path.basename(os.path.normpath(input_path))
        out_dir = os.path.join(input_path, f"{folder_name}_output")
        ensure_dir(out_dir)

        main_csv = os.path.join(out_dir, f"{folder_name}.csv")
        header = [
            "seed", "Designed_seq", "E_fine(kBT)", "recovery", "macroF1", "PPL",
            "div_ensemble", "T_min", "count"
        ]

        print(f"[INFO] Multi-state folder: {input_path}")
        print(f"[INFO] Output folder      : {out_dir}")
        print(f"[INFO] Fail log          : {_fail_log_path(out_dir)}")

        # Fetch reference PDB for multi-state auto-parsing
        pdb_paths = glob.glob(os.path.join(input_path, "*.pdb"))
        pdb_paths = _sort_multistate_pdbs_target_first(pdb_paths)
        ref_pdb = pdb_paths[0] if pdb_paths else None

        thermo_ss_arg, hard_ss_arg = _resolve_design_ss_dual_multi(
            ss_arg_cli,
            input_path,
            ref_pdb,
        )
        _print_design_ss_preview(
            f"multi:{folder_name}",
            thermo_ss_arg,
            hard_ss_arg,
            pdb_path=ref_pdb,
            original_ss_arg=("auto" if ss_arg_cli is None else ss_arg_cli),
        )
        chain_break_info = chain_break_info_from_ss(thermo_ss_arg or hard_ss_arg)
        frz_arg_this = _resolve_frz_for_multi(frz_arg_cli, input_path)
        _print_frz_preview(f"multi:{folder_name}", frz_arg_this)

        repeat_runs_total = int(args.batch)
        reproduce_seed = args.seed
        reproduce_seed_batch = None if args.seed_batch is None else read_seed_batch_file(args.seed_batch)
        batch_size = int(args.batch_size)
        kl_lambda = float(args.kl_lambda)
        T_min = None if args.temp is None else float(args.temp)
        ss_penalty = float(args.ss_penalty)

        def _run_fn(seeds: List[int]) -> List[Dict[str, Any]]:
            return run_multi_folder_multi(
                multi_dir=input_path,
                seeds=seeds,
                pot=pot,
                device=device,
                steps=steps,
                batch_size=batch_size,
                tail_steps=tail_steps,
                kl_lambda=kl_lambda,
                T_min=T_min,
                thermo_ss_arg=thermo_ss_arg,
                hard_ss_arg=hard_ss_arg,
                ss_penalty=ss_penalty,
                frz_arg=frz_arg_this,
            )

        def _handle_one(seed: int, res: Dict[str, Any]) -> bool:
            tag = f"multi:{folder_name}"
            try:
                if args.debug_traj:
                    _debug_print(res, tag=f"{tag}:seed={seed}")

                Ensemble_E, _, _ = ensemble_avg_energy_from_tail(
                    tail_seq_u8_list=res.get("tail_seq_u8"),
                    tail_E_list=res.get("tail_E"),
                    beta=ENSEMBLE_BETA,
                    energy_mode="min",
                )
                E_fine_DS = compute_E_fine_DS(res["E_cons"], Ensemble_E, res["E_rough"])
                consensus_disp = seq_display(res["consensus_seq"], mol_mode, chain_break_info)
                native_idx = res.get("native_idx", None)

                if has_masked_sites(native_idx):
                    n_mask = count_masked_sites(native_idx)
                    print(
                        f"[INFO] {folder_name} | seed={seed}: detected {n_mask} unconstrained target-native site(s). "
                        f"These positions are shown as '-' in Ensemble/traj exports and are excluded from masked metrics."
                    )

                write_or_merge_main_csv(
                    main_csv,
                    [
                        seed,
                        consensus_disp,
                        f"{E_fine_DS:.6f}",
                        f"{res['rec']:.6f}",
                        f"{res['f1']:.6f}",
                        f"{res['ppl']:.6f}",
                        f"{res['div']:.6f}",
                        ("" if T_min is None else f"{T_min:.6f}"),
                        1,
                    ],
                    header=header,
                )

                _export_topE_for_one_seed(
                    out_dir=out_dir,
                    topE=topE,
                    seed=seed,
                    res=res,
                    mol=mol_mode,
                    chain_break_info=chain_break_info,
                )
                _export_ensemble_for_one_seed(
                    out_dir=out_dir,
                    seed=seed,
                    res=res,
                    mol=mol_mode,
                    chain_break_info=chain_break_info,
                )

                pretty_print_run(
                    folder_name,
                    seed,
                    consensus_disp,
                    res["rec"],
                    res["f1"],
                    res["ppl"],
                    res["div"],
                    E_fine_DS,
                    topE,
                    out_dir,
                )
                return True
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                append_fail_log(out_dir, target_tag=tag, seed=int(seed), stage="postprocess", exc=e)
                print(f"[WARN] Postprocess failed for seed={seed}. Logged to fail.log. Continue.")
                return False
            finally:
                _drop_big_fields_inplace(res)

        def _round_seed_writer(round_idx: int, seeds: List[int]):
            write_seed_batch_file(
                out_dir=out_dir,
                target_tag=f"multi:{folder_name}",
                round_idx=round_idx,
                seeds=seeds,
            )

        ok = 0
        try:
            if reproduce_seed is not None:
                ok = run_exact_one_seed(
                    seed=reproduce_seed,
                    run_fn=_run_fn,
                    handle_one=_handle_one,
                    tag=f"multi:{folder_name}",
                    out_dir=out_dir,
                )
            elif reproduce_seed_batch is not None:
                ok = run_exact_seed_batch(
                    seeds=reproduce_seed_batch,
                    run_fn=_run_fn,
                    handle_one=_handle_one,
                    tag=f"multi:{folder_name}",
                    out_dir=out_dir,
                )
            else:
                ok = run_streaming_collect_success(
                    target_total=repeat_runs_total,
                    run_fn=_run_fn,
                    handle_one=_handle_one,
                    tag=f"multi:{folder_name}",
                    out_dir=out_dir,
                    round_seed_writer=_round_seed_writer,
                )
        finally:
            _cleanup_cuda()

        if reproduce_seed is not None:
            print(f"[INFO] Exact-seed run done for {folder_name}. Successful runs = {ok}/1. Summary CSV: {main_csv}")
        elif reproduce_seed_batch is not None:
            print(f"[INFO] Exact seed-batch replay done for {folder_name}. Successful runs = {ok}/{len(reproduce_seed_batch)}. Summary CSV: {main_csv}")
        else:
            print(f"[INFO] Done for {folder_name}. Successful runs = {ok}/{repeat_runs_total}. Summary CSV: {main_csv}")
        print("[INFO] All done.")
        _cleanup_cuda()
        return

    is_single_input = os.path.isfile(input_path)
    is_batch_input = os.path.isdir(input_path)

    if is_single_input:
        pdb_list = [input_path]
    elif is_batch_input:
        pdb_list = sorted(glob.glob(os.path.join(input_path, "*.pdb")))
        if not pdb_list:
            raise RuntimeError(f"[ERROR] No .pdb files found under: {input_path}")
    else:
        raise RuntimeError(f"[ERROR] Input is neither file nor directory: {input_path}")

    print(f"[INFO] Targets: {len(pdb_list)}")

    for pdb_path in pdb_list:
        base = pdb_base_name(pdb_path)
        out_dir = os.path.join(os.path.dirname(pdb_path), f"{base}_output")
        ensure_dir(out_dir)

        main_csv = os.path.join(out_dir, f"{base}.csv")
        header = [
            "seed", "Designed_seq", "E_fine(kBT)", "recovery", "macroF1", "PPL",
            "div_ensemble", "T_min", "count"
        ]

        print(f"[INFO] Target: {base}")
        print(f"[INFO] Output: {out_dir}")
        print(f"[INFO] Fail log: {_fail_log_path(out_dir)}")

        if is_batch_input:
            thermo_ss_arg, hard_ss_arg = _resolve_design_ss_dual_batch(ss_arg_cli, pdb_path)
            frz_arg_this = _resolve_frz_for_batch(frz_arg_cli, pdb_path)
        else:
            thermo_ss_arg, hard_ss_arg = _resolve_design_ss_dual_single(ss_arg_cli, pdb_path)
            frz_arg_this = _resolve_frz_for_single(frz_arg_cli, pdb_path)

        _print_design_ss_preview(
            f"single:{base}",
            thermo_ss_arg,
            hard_ss_arg,
            pdb_path=pdb_path,
            original_ss_arg=("auto" if ss_arg_cli is None else ss_arg_cli),
        )
        chain_break_info = chain_break_info_from_ss(thermo_ss_arg or hard_ss_arg)
        _print_frz_preview(f"single:{base}", frz_arg_this)

        def _run_fn(seeds: List[int]) -> List[Dict[str, Any]]:
            # For a single PDB with exactly one seed, force the true single-path
            # execution so that --batch 1 / exact replay matches direct single-run
            # behavior as strictly as possible.
            if len(seeds) == 1:
                s = int(seeds[0])
                set_seed(s)
                return [run_single_pdb_once(
                    pdb_path=pdb_path,
                    pot=pot,
                    device=device,
                    steps=steps,
                    batch_size=batch_size,
                    tail_steps=tail_steps,
                    kl_lambda=kl_lambda,
                    T_min=T_min,
                    thermo_ss_arg=thermo_ss_arg,
                    hard_ss_arg=hard_ss_arg,
                    ss_penalty=ss_penalty,
                    frz_arg=frz_arg_this,
                )]

            return run_single_pdb_multi(
                pdb_path=pdb_path,
                seeds=seeds,
                pot=pot,
                device=device,
                steps=steps,
                batch_size=batch_size,
                tail_steps=tail_steps,
                kl_lambda=kl_lambda,
                T_min=T_min,
                thermo_ss_arg=thermo_ss_arg,
                hard_ss_arg=hard_ss_arg,
                ss_penalty=ss_penalty,
                frz_arg=frz_arg_this,
            )

        def _handle_one(seed: int, res: Dict[str, Any]) -> bool:
            tag = f"single:{base}"
            try:
                if args.debug_traj:
                    _debug_print(res, tag=f"{tag}:seed={seed}")

                Ensemble_E, _, _ = ensemble_avg_energy_from_tail(
                    tail_seq_u8_list=res.get("tail_seq_u8"),
                    tail_E_list=res.get("tail_E"),
                    beta=ENSEMBLE_BETA,
                    energy_mode="min",
                )
                E_fine_DS = compute_E_fine_DS(res["E_cons"], Ensemble_E, res["E_rough"])
                consensus_disp = seq_display(res["consensus_seq"], mol_mode, chain_break_info)
                native_idx = res.get("native_idx", None)

                if has_masked_sites(native_idx):
                    n_mask = count_masked_sites(native_idx)
                    print(
                        f"[INFO] {base} | seed={seed}: detected {n_mask} unconstrained target-native site(s). "
                        f"These positions are shown as '-' in Ensemble/traj exports and are excluded from masked metrics."
                    )

                write_or_merge_main_csv(
                    main_csv,
                    [
                        seed,
                        consensus_disp,
                        f"{E_fine_DS:.6f}",
                        f"{res['rec']:.6f}",
                        f"{res['f1']:.6f}",
                        f"{res['ppl']:.6f}",
                        f"{res['div']:.6f}",
                        ("" if T_min is None else f"{T_min:.6f}"),
                        1,
                    ],
                    header=header,
                )

                _export_topE_for_one_seed(
                    out_dir=out_dir,
                    topE=topE,
                    seed=seed,
                    res=res,
                    mol=mol_mode,
                    chain_break_info=chain_break_info,
                )
                _export_ensemble_for_one_seed(
                    out_dir=out_dir,
                    seed=seed,
                    res=res,
                    mol=mol_mode,
                    chain_break_info=chain_break_info,
                )

                pretty_print_run(
                    base,
                    seed,
                    consensus_disp,
                    res["rec"],
                    res["f1"],
                    res["ppl"],
                    res["div"],
                    E_fine_DS,
                    topE,
                    out_dir,
                )
                return True
            except KeyboardInterrupt:
                raise
            except BaseException as e:
                append_fail_log(out_dir, target_tag=tag, seed=int(seed), stage="postprocess", exc=e)
                print(f"[WARN] Postprocess failed for seed={seed}. Logged to fail.log. Continue.")
                return False
            finally:
                _drop_big_fields_inplace(res)

        def _round_seed_writer(round_idx: int, seeds: List[int]):
            write_seed_batch_file(
                out_dir=out_dir,
                target_tag=f"single:{base}",
                round_idx=round_idx,
                seeds=seeds,
            )

        ok = 0
        try:
            if reproduce_seed is not None:
                ok = run_exact_one_seed(
                    seed=reproduce_seed,
                    run_fn=_run_fn,
                    handle_one=_handle_one,
                    tag=f"single:{base}",
                    out_dir=out_dir,
                )
            elif reproduce_seed_batch is not None:
                ok = run_exact_seed_batch(
                    seeds=reproduce_seed_batch,
                    run_fn=_run_fn,
                    handle_one=_handle_one,
                    tag=f"single:{base}",
                    out_dir=out_dir,
                )
            else:
                ok = run_streaming_collect_success(
                    target_total=repeat_runs_total,
                    run_fn=_run_fn,
                    handle_one=_handle_one,
                    tag=f"single:{base}",
                    out_dir=out_dir,
                    round_seed_writer=_round_seed_writer,
                )
        finally:
            _cleanup_cuda()

        if reproduce_seed is not None:
            print(f"[INFO] Exact-seed run done for {base}. Successful runs = {ok}/1. Summary CSV: {main_csv}")
        elif reproduce_seed_batch is not None:
            print(f"[INFO] Exact seed-batch replay done for {base}. Successful runs = {ok}/{len(reproduce_seed_batch)}. Summary CSV: {main_csv}")
        else:
            print(f"[INFO] Done for {base}. Successful runs = {ok}/{repeat_runs_total}. Summary CSV: {main_csv}")

    print("[INFO] All done.")
    _cleanup_cuda()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C). Exiting now.")
        _cleanup_cuda()
        raise SystemExit(130)
