# Single-state design example

This example uses the 36-residue scaffold `inputs/8VY0.pdb`.

## Inputs

- `inputs/8VY0.pdb`: target 3D scaffold;
- `inputs/8VY0.dbn`: six-pair secondary structure derived with DSSR;
- `inputs/no_contact.dbn`: same-length structure with no paired positions.

## Run with automatic secondary structure

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb --batch 10
```

This uses the internal sequence-independent parser for the local thermodynamic screen and does not impose a hard base-pair penalty.

## Run with the supplied DBN constraint

```bash
python DS3dRNA.py Examples/Design/inputs/8VY0.pdb \
  --ss Examples/Design/inputs/8VY0.dbn \
  --batch 10
```

Explicit `--ss` activates both thermodynamic and hard secondary-structure constraints and uses the fixed constrained trajectory parameters documented in [the quick-start guide](../../docs/QUICKSTART.md).

## Post-process completed runs

Convert summary CSV files into AlphaFold 3 JSON inputs:

```bash
python Examples/Design/DS3dRNA_consensus_script.py \
  -i path/to/results \
  -o path/to/af3_json \
  --count --max_run 5
```

Aggregate metrics from result subdirectories:

```bash
cd path/to/results
python /path/to/DS3dRNA/Examples/Design/summarize_weighted_by_count_deltaE10000.py
```

The latter writes `sum.csv` and `sum_top1.csv` in the current directory.
