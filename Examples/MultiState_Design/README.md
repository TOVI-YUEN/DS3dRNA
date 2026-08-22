# Multi-state design example

This example jointly conditions one sequence on five 78-residue, two-chain conformations.

## Inputs

- `inputs/fold_es2_env_9_derivative_model_0.pdb` through `_4.pdb`: residue-matched conformations;
- `inputs/ss.dbn`: two-chain dot-bracket constraint, including pseudoknot brackets;
- `inputs/Frz.fasta`: frozen mask that preserves the second chain while leaving the first chain designable.

## Run

```bash
python DS3dRNA.py -m Examples/MultiState_Design/inputs \
  --ss Examples/MultiState_Design/inputs/ss.dbn \
  --frz Examples/MultiState_Design/inputs/Frz.fasta \
  --batch 10
```

The PDB files must remain residue-matched and in a consistent chain/residue order. The `&` characters in the DBN and frozen mask mark the same chain boundary and are not counted as residues.

The post-processing scripts in this directory use the same interface described for the [single-state example](../Design/README.md).
