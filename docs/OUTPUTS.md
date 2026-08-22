# Outputs

[← DS3dRNA](../README.md) · [Quick start](QUICKSTART.md) · [Inputs](INPUTS.md)

## Design output directory

For a target named `target.pdb`, DS3dRNA creates `target_output/` next to the input structure. A completed single-seed run normally contains:

- `target.csv`: one-row run summary;
- `Ensemble_<seed>.csv.zip`: compressed accepted-sequence ensemble;
- `trajTopE_<K>_<seed>.csv.zip`: compressed unique sequences with the lowest observed Fine energies across the trajectory;
- `fail.log`: created or updated when a run fails.

Multi-run and multi-state jobs add seed-specific or multi-state naming while retaining the same information categories.

## Summary fields

The summary CSV reports the designed sequence and diagnostics such as:

- `Designed_seq`: decoded design sequence;
- `E_fine(kBT)`: Fine TriRNASP energy;
- `recovery`: identity to the sequence encoded in the input scaffold, when available;
- `macroF1`: nucleotide-class-balanced agreement to the encoded sequence;
- `PPL`: profile perplexity diagnostic;
- `div_ensemble`: ensemble diversity diagnostic;
- `seed`: random seed used for the run.

Recovery and MacroF1 are diagnostics against input residue identities, not optimization targets and not experimental validation.

## Rank output

DS3dRank writes CSV rows containing at least:

- `sequence`;
- `energy(kBT)`;
- `recovery`;
- `macroF1`.

Rows are ordered by Fine energy, with lower energy ranked first.

## Post-processing

The design examples include two utilities:

- `DS3dRNA_consensus_script.py`: creates count-weighted consensus and minimum-Fine-energy AlphaFold 3 input JSON files;
- `summarize_weighted_by_count_deltaE10000.py`: aggregates result CSV metrics using `count` as a frequency weight and excludes rows more than 10,000 kBT above the within-file Fine-energy minimum.

These scripts are analysis helpers, not part of the sampler. See the [single-state](../Examples/Design/README.md) and [multi-state](../Examples/MultiState_Design/README.md) example guides.
