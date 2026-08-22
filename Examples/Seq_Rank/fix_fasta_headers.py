#!/usr/bin/env python3
"""Add missing FASTA headers to one-sequence-per-line files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def fix_fasta(input_path: Path, output_path: Path, prefix: str) -> int:
    """Write valid FASTA records and return the number of sequences written."""
    lines = input_path.read_text(encoding="utf-8").splitlines()
    records: list[tuple[str, str]] = []
    pending_header: str | None = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line_number == 1 and line.lower() == "sequence":
            continue
        if line.startswith(">"):
            pending_header = line
            continue

        header = pending_header or f">{prefix}_{len(records) + 1:04d}"
        records.append((header, line))
        pending_header = None

    if pending_header is not None:
        raise ValueError(f"Header without a sequence: {pending_header}")
    if not records:
        raise ValueError(f"No sequences found in {input_path}")

    output_text = "".join(f"{header}\n{sequence}\n" for header, sequence in records)
    output_path.write_text(output_text, encoding="utf-8")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a unique FASTA header before every sequence."
    )
    parser.add_argument("input", type=Path, help="Input sequence/FASTA file")
    parser.add_argument(
        "output", type=Path, nargs="?", help="Output FASTA file (omit with --in-place)"
    )
    parser.add_argument("--prefix", default="candidate", help="Header prefix")
    parser.add_argument(
        "--in-place", action="store_true", help="Replace input and save input.bak"
    )
    args = parser.parse_args()

    if args.in_place and args.output is not None:
        parser.error("output cannot be used together with --in-place")
    if not args.in_place and args.output is None:
        parser.error("output is required unless --in-place is used")

    input_path = args.input.resolve()
    if args.in_place:
        backup_path = input_path.with_name(input_path.name + ".bak")
        temp_path = input_path.with_name(input_path.name + ".tmp")
        shutil.copy2(input_path, backup_path)
        try:
            count = fix_fasta(input_path, temp_path, args.prefix)
            temp_path.replace(input_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        print(f"Wrote {count} records to {input_path} (backup: {backup_path})")
    else:
        output_path = args.output.resolve()
        count = fix_fasta(input_path, output_path, args.prefix)
        print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
