#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Summarize DS3dRNA result CSV files with count-weighted statistics."""

import csv
import math
from pathlib import Path

# Metrics to summarize.
METRICS = ["recovery", "macroF1", "PPL", "div_ensemble"]

# Energy column used to select the top row (lower is better).
TOP1_KEY = "E_fine(kBT)"

# Exclude rows for which E_fine - E_fine_min exceeds this threshold.
# Set to None to disable the filter.
DELTA_E_MAX = 10000.0

ENERGY_CANDIDATES = [TOP1_KEY, "E_fine_kBT", "E_fine", "Efine", "fine"]

# Count-weight column. Rows default to a weight of 1 when it is absent.
COUNT_KEY = "count"

# Search nested subdirectories when enabled.
RECURSIVE = False

# Output files.
SUM_OUT = "sum.csv"
TOP1_OUT = "sum_top1.csv"


def _norm_key(x) -> str:
    """Normalize minor case and punctuation differences in column names."""
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )


def build_field_map(fieldnames):
    return {
        _norm_key(x): x
        for x in (fieldnames or [])
        if x is not None and str(x).strip()
    }


def get_field(row, field_map, candidates, default=""):
    for c in candidates:
        k = _norm_key(c)
        if k in field_map:
            return row.get(field_map[k], default)
    return default


def safe_float(x):
    try:
        v = float(str(x).strip())
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def safe_weight(x):
    """Return a positive count weight, or None for an unusable value."""
    v = safe_float(x)
    if v is None or v <= 0:
        return None
    return v


def weighted_mean_std(pairs):
    """Return the weighted mean, population standard deviation, and weight.

    ``pairs`` contains ``(value, weight)`` entries. The calculation is:

        mean = sum(w*x) / sum(w)
        std  = sqrt(sum(w*(x-mean)^2) / sum(w))

    Population standard deviation is used because ``count`` represents the
    frequency of each row in a compressed unique-sequence table.
    """
    pairs = [(v, w) for v, w in pairs if v is not None and w is not None and w > 0]
    if not pairs:
        return "nan", "nan", 0.0

    sw = sum(w for _, w in pairs)
    if sw <= 0:
        return "nan", "nan", 0.0

    mean = sum(v * w for v, w in pairs) / sw
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / sw
    std = math.sqrt(max(var, 0.0))
    return mean, std, sw


def iter_target_csv_files(root=Path(".")):
    """Yield result CSV files, excluding trajectory and generated summaries."""
    if RECURSIVE:
        candidates = root.rglob("*.csv")
    else:
        candidates = []
        for subdir in sorted(p for p in root.iterdir() if p.is_dir()):
            candidates.extend(sorted(subdir.glob("*.csv")))

    for csv_file in sorted(candidates):
        name = csv_file.name
        name_lower = name.lower()
        if name_lower.startswith("ensemble") or name_lower.startswith("traj"):
            continue
        if name in {SUM_OUT, TOP1_OUT}:
            continue
        yield csv_file


out_rows = []
top1_rows = []

for csv_file in iter_target_csv_files(Path(".")):
    # Store (value, count_weight) pairs for sum.csv.
    weighted_values = {m: [] for m in METRICS}

    # Retain the lowest-Fine-energy row for sum_top1.csv.
    best_row = None
    best_escore = None
    best_count = None

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        field_map = build_field_map(reader.fieldnames)
        rows = list(reader)

    # Find the minimum Fine energy within this CSV.
    energy_values = []
    for row in rows:
        escore = safe_float(get_field(row, field_map, ENERGY_CANDIDATES, default=""))
        if escore is not None:
            energy_values.append(escore)

    e_fine_min = min(energy_values) if energy_values else None
    n_rows_before_filter = len(rows)
    n_rows_after_filter = 0
    n_rows_deltaE_filtered = 0

    for row in rows:
        escore = safe_float(get_field(row, field_map, ENERGY_CANDIDATES, default=""))

        # Exclude high-energy rows. If the CSV has no valid Fine energy, no
        # delta-E filter is applied.
        if DELTA_E_MAX is not None and e_fine_min is not None:
            if escore is None or (escore - e_fine_min) > DELTA_E_MAX:
                n_rows_deltaE_filtered += 1
                continue

        n_rows_after_filter += 1

        # Default to unit weight when no count column is available.
        count_raw = get_field(row, field_map, [COUNT_KEY, "Count", "n", "freq", "frequency"], default="1")
        w = safe_weight(count_raw)

        # Accumulate count-weighted metrics for sum.csv.
        if w is not None:
            for m in METRICS:
                v = safe_float(get_field(row, field_map, [m], default=""))
                if v is not None:
                    weighted_values[m].append((v, w))

        # Track the lowest-Fine-energy row for sum_top1.csv.
        if escore is not None:
            if best_escore is None or escore < best_escore:
                best_escore = escore
                best_row = row.copy()
                best_count = w

    # Skip CSV files that contain no usable metrics.
    if all(len(weighted_values[m]) == 0 for m in METRICS):
        continue

    # Build the sum.csv row.
    row_out = {"pdb": csv_file.stem}
    row_out["E_fine_min"] = e_fine_min if e_fine_min is not None else "nan"
    row_out["rows_before_filter"] = n_rows_before_filter
    row_out["rows_after_filter"] = n_rows_after_filter
    row_out["rows_deltaE_filtered"] = n_rows_deltaE_filtered
    total_count_candidates = []

    for m in METRICS:
        mean, std, sw = weighted_mean_std(weighted_values[m])
        row_out[f"{m}_mean"] = mean
        row_out[f"{m}_std"] = std
        total_count_candidates.append(sw)

    # Record total weight for effective-sample-size checks.
    valid_counts = [x for x in total_count_candidates if x and x > 0]
    row_out["total_count"] = max(valid_counts) if valid_counts else 0.0

    out_rows.append(row_out)

    # Build the sum_top1.csv row.
    if best_row is not None:
        top1_out = {
            "pdb": csv_file.stem,
            "E_score": best_escore,
            "E_fine_min": e_fine_min if e_fine_min is not None else "nan",
            "count": best_count if best_count is not None else "nan",
            "rows_before_filter": n_rows_before_filter,
            "rows_after_filter": n_rows_after_filter,
            "rows_deltaE_filtered": n_rows_deltaE_filtered,
        }
        for m in METRICS:
            v = safe_float(get_field(best_row, build_field_map(best_row.keys()), [m], default=""))
            top1_out[m] = v if v is not None else "nan"
        top1_rows.append(top1_out)


# Write sum.csv.
out_file = Path(SUM_OUT)
with out_file.open("w", encoding="utf-8", newline="") as f:
    fieldnames = ["pdb", "E_fine_min", "rows_before_filter", "rows_after_filter", "rows_deltaE_filtered"]
    for m in METRICS:
        fieldnames += [f"{m}_mean", f"{m}_std"]
    fieldnames += ["total_count"]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(out_rows)


# Write sum_top1.csv.
top1_file = Path(TOP1_OUT)
with top1_file.open("w", encoding="utf-8", newline="") as f:
    fieldnames = ["pdb", "E_score", "E_fine_min", "count", "rows_before_filter", "rows_after_filter", "rows_deltaE_filtered"] + METRICS
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(top1_rows)


print(f"[OK] Wrote weighted summary for {len(out_rows)} PDB files -> {out_file}")
print(f"[OK] Wrote top1 summary for {len(top1_rows)} PDB files -> {top1_file}")
print("[INFO] sum.csv uses count-weighted mean/std after filtering rows with E_fine - E_fine_min > 10000.")
print("[INFO] sum_top1.csv selects the minimum E_fine(kBT) row after the same filtering rule.")
