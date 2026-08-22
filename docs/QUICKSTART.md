# Quick start

[← DS3dRNA](../README.md) · [Inputs](INPUTS.md) · [Outputs](OUTPUTS.md)

Run all commands from the repository root after activating the DS3dRNA environment.

## 1. Single-state design

With automatic secondary-structure inference for thermodynamic screening:

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb --batch 10
```

With an explicit dot-bracket constraint:

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb \
  --ss Examples/Design/inputs/8VY0.dbn \
  --batch 10
```

Important: in design mode, an explicit `--ss` value (including `--ss auto` and `--ss none`) activates the hard secondary-structure path and fixes `steps=10000`, `tail_steps=4000`, and `topE=10000`. Omitted `--ss` still supplies automatically inferred secondary structure to the local thermodynamic screen, but does not enable the hard base-pair penalty. `--ss none` explicitly supplies an all-unpaired, zero-contact DBN; see [Inputs and constraints](INPUTS.md#secondary-structure) for the complete distinction.

## 2. Reproduce a run

Run exactly one seed:

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb --seed 363022884
```

Replay an ordered list of seeds, one integer per line:

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb \
  --seed_batch seed_list.txt
```

When `--seed` or `--seed_batch` is supplied, `--batch` is ignored.

## 3. Batch design

Pass a directory containing PDB targets:

```bash
python DS3dRNA.py path/to/pdb_directory --batch 10
```

For per-target DBN or frozen files, pass a directory through `--ss` or `--frz`. Files are matched by PDB basename; see [INPUTS.md](INPUTS.md).

## 4. Multi-state design

All PDB files in the multi-state folder must describe residue-matched conformations of the same molecule.

```bash
python DS3dRNA.py -m Examples/MultiState_Design/inputs \
  --ss Examples/MultiState_Design/inputs/ss.dbn \
  --frz Examples/MultiState_Design/inputs/Frz.fasta \
  --batch 10
```

## 5. Sequence ranking with DS3dRank

Single scaffold:

```bash
python DS3dRNA.py -rank \
  --str 'Examples/Seq_Rank/R1138_7PTL_A:720.pdb' \
  --fa Examples/Seq_Rank/candidates.fasta \
  --rank_out ranked_candidates.csv
```

Directory of independent scaffolds:

```bash
python DS3dRNA.py -rank --batch_rank \
  --str path/to/pdb_directory \
  --fa candidates.fasta \
  --rank_out rank_outputs
```

One sequence set scored jointly against a multi-state directory:

```bash
python DS3dRNA.py -rank --multi_rank \
  --str path/to/multi_state_directory \
  --fa candidates.fasta \
  --rank_out multi_state.rank.csv
```

Rank mode uses the Fine three-body energy only. Thermodynamic reranking is disabled; `--ss` is optional and acts only as a hard constraint filter.

## 6. RNA and DNA modes

RNA is the default. Select the DNA energy tensors and DNA thermodynamic backend with:

```bash
python DS3dRNA.py target.pdb --mol DNA --batch 10
```

DNA mode displays T rather than U and limits paired constraints to AT/TA/CG/GC.

The same selector applies to ranking:

```bash
python DS3dRNA.py -rank \
  --str target_DNA.pdb \
  --fa DNA_candidates.fasta \
  --mol DNA \
  --rank_out DNA_candidates.rank.csv
```

See [Inputs and constraints](INPUTS.md#rna-and-dna-molecule-modes) for the complete RNA/DNA comparison.

## 7. Results

Each target receives its own output directory containing a summary CSV and compressed ensemble/trajectory files. See [OUTPUTS.md](OUTPUTS.md) for field definitions and post-processing utilities.
