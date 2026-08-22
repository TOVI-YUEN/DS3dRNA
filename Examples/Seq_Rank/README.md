# Sequence-ranking example

This example demonstrates DS3dRank on a 720-residue scaffold.

## Files

- `R1138_7PTL_A:720.pdb`: fixed 3D scaffold;
- `candidates.fasta`: candidate sequences with unique FASTA headers;
- `trajTopE_10000_18625168.csv`: source trajectory table from which candidates were prepared;
- `R1138_7PTL_A:720.rank.csv`: bundled reference ranking output;
- `fix_fasta_headers.py`: utility for converting one-sequence-per-line text into valid FASTA.

The colon in the example filename is valid on Linux. The bundled installer and documented workflow target Linux.

## Rank candidates

Write a new result without overwriting the bundled reference CSV:

```bash
python DS3dRNA.py -rank \
  --str 'Examples/Seq_Rank/R1138_7PTL_A:720.pdb' \
  --fa Examples/Seq_Rank/candidates.fasta \
  --rank_out seq_rank_check.csv
```

Use `--rank_bs` to control the scoring batch size. Lower values reduce peak memory use.

## Repair a headerless sequence file

```bash
python Examples/Seq_Rank/fix_fasta_headers.py \
  raw_sequences.txt candidates.fasta --prefix candidate
```

Add `--in-place` instead of an output path to replace the input and create an `.bak` backup.
