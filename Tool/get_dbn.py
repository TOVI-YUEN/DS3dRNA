#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch DSSR runner:
  recursively find PDB files -> run DSSR -> save original dssr-2ndstrs.dbn

Features:
  - Recursive PDB discovery: pdb_dir/**/*.pdb
  - Unique output stem from relative path (avoid overwrite)
  - Save original DSSR dbn file WITHOUT any modification
  - Save DSSR stdout/stderr to .dssr.log
  - Record failures to fail_list.txt

Example:
  python Tool/get_dbn.py \
    --pdb_dir ./T136 \
    --dssr_bin ./x3dna-dssr \
    --out_dbn_dir ./dbn_out
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_dssr_on_pdb(dssr_bin: Path, pdb_path: Path, out_dbn_path: Path, log_path: Path) -> None:
    """
    Run DSSR in a temporary directory and copy the original DSSR-generated
    dssr-2ndstrs.dbn to out_dbn_path WITHOUT any modification.
    """
    out_dbn_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        proc = subprocess.run(
            [str(dssr_bin), f"-i={pdb_path}"],
            cwd=td,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        log_path.write_text(proc.stdout, encoding="utf-8", errors="ignore")

        dbn_src = td / "dssr-2ndstrs.dbn"
        if not dbn_src.exists():
            raise RuntimeError(f"No dssr-2ndstrs.dbn generated. See log: {log_path}")

        # Preserve DSSR output exactly as produced.
        shutil.copyfile(dbn_src, out_dbn_path)


def main():
    ap = argparse.ArgumentParser(
        description="Batch-run DSSR on PDB files and preserve its .dbn outputs."
    )
    ap.add_argument(
        "--pdb_dir", required=True, help="Directory containing PDB files (recursive search)"
    )
    ap.add_argument("--dssr_bin", required=True, help="Path to the x3dna-dssr executable")
    ap.add_argument(
        "--out_dbn_dir", default="dbn_out", help="Directory for .dbn files and DSSR logs"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Maximum number of PDBs to process (0 = no limit)"
    )
    ap.add_argument(
        "--fail_list", default="fail_list.txt", help="File in which failed items are recorded"
    )
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir).expanduser().resolve()
    dssr_bin = Path(args.dssr_bin).expanduser().resolve()
    out_dbn_dir = Path(args.out_dbn_dir).expanduser().resolve()
    fail_path = Path(args.fail_list).expanduser().resolve()

    if not pdb_dir.is_dir():
        raise SystemExit(f"[ERROR] pdb_dir not found: {pdb_dir}")
    if not dssr_bin.exists():
        raise SystemExit(f"[ERROR] dssr_bin not found: {dssr_bin}")
    if not os.access(dssr_bin, os.X_OK):
        raise SystemExit(f"[ERROR] dssr_bin is not executable: {dssr_bin}")

    pdbs = sorted(pdb_dir.rglob("*.pdb"))
    if args.limit and args.limit > 0:
        pdbs = pdbs[:args.limit]

    print(f"[INFO] Found {len(pdbs)} PDB files in: {pdb_dir}")
    print(f"[INFO] DSSR binary: {dssr_bin}")
    print(f"[INFO] Output dir: {out_dbn_dir}")
    print(f"[INFO] Fail list: {fail_path}")

    fail_path.parent.mkdir(parents=True, exist_ok=True)
    if not fail_path.exists():
        fail_path.write_text("", encoding="utf-8")

    n_ok = 0
    n_fail = 0

    for pdb_path in pdbs:
        rel = pdb_path.relative_to(pdb_dir)
        stem = rel.with_suffix("").as_posix().replace("/", "__")

        dbn_out = out_dbn_dir / f"{stem}.dbn"
        log_out = out_dbn_dir / f"{stem}.dssr.log"

        print("\n============================================================")
        print(f"[RUN] {pdb_path}")
        print(f"[OUT] {dbn_out}")

        try:
            run_dssr_on_pdb(dssr_bin, pdb_path, dbn_out, log_out)
            print(f"[OK] {dbn_out}")
            n_ok += 1
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {pdb_path} -> {reason}")
            with open(fail_path, "a", encoding="utf-8") as f:
                f.write(f"{stem}\t{pdb_path}\t{reason}\tlog={log_out}\n")
            n_fail += 1

    print("\n============================================================")
    print(f"[DONE] success={n_ok}, failed={n_fail}")


if __name__ == "__main__":
    main()
