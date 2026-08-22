# Method overview

[← DS3dRNA](../README.md) · [Inputs](INPUTS.md) · [Citations](../CITATIONS.md)

This page describes the scientific workflow exposed by `DS3dRNA.py`. It intentionally avoids internal module-level implementation details.

## Design objective

DS3dRNA addresses fixed-backbone inverse design: given one or more target 3D nucleic-acid scaffolds, search for a sequence that stabilizes the target geometry while satisfying optional secondary-structure and frozen-site constraints. Single-state design uses one scaffold; multi-state design optimizes one sequence jointly against residue-matched conformations.

## Coarse-grained structural representation

Each residue is represented by three sites:

- P and C4′ for the backbone;
- N1 for pyrimidines or N9 for purines as the base anchor.

This representation preserves the geometry needed to construct higher-order contacts without supplying a complete nucleotide-specific heavy-atom pattern to the design model.

## Higher-order interaction energy

DS3dRNA uses the three-body framework developed for TriRNASP. Unique atom triplets are enumerated from the coarse-grained scaffold and scored through two tabulated potentials:

- Rough energy for broad ensemble evaluation;
- Fine energy for proposal selection and final ranking.

The RNA Rough and Fine spatial discretizations correspond to 7 × 7 × 14 and 18 × 18 × 36 distance-bin grids, respectively, across nucleotide-specific coarse-grained atom types. Distributed tensors are described in [Energy/README.md](../Energy/README.md).

## Monte Carlo sequence generation

The default workflow initializes an ensemble of 320 sequences at unconstrained positions. At each step:

1. Rough energies define Boltzmann-weighted nucleotide profiles at every position.
2. New candidates are sampled from those profiles and combined with low-Rough-energy ensemble members.
3. A local nearest-neighbor thermodynamic model selects five candidates.
4. Those candidates are rescored with the Fine potential.
5. The lowest-Fine-energy proposal is accepted or rejected with a Metropolis update.
6. Mutation probability is adapted from recent acceptance behavior.

The annealing coefficient decreases toward a default final value of 4.5 kBT. A standard constrained trajectory uses 10,000 steps, and the final 40% of accepted states contributes to the output nucleotide profile.

## Thermodynamic screening

The public `--mol RNA|DNA` selector chooses the molecule-specific energy tensors, alphabet, and thermodynamic backend. RNA mode uses local terms from the Turner 2004 nearest-neighbor model. DNA mode uses the corresponding local enthalpy and entropy tables distributed with Primer3 and based on the SantaLucia nearest-neighbor framework. Command examples and the complete mode comparison are provided in [Inputs and constraints](INPUTS.md#rna-and-dna-molecule-modes).

The screen retains local stacking, hairpin, bulge, and supported internal-loop terms. Global-context contributions such as multibranch, exterior, coaxial-stacking, and dangling-end terms are not included in this restricted mid-stage screen. This thermodynamic score filters proposals; it does not replace the three-body design energy.

## Constraints and decoding

Secondary-structure constraints restrict paired mutations to canonical and wobble combinations in RNA mode and Watson-Crick combinations in DNA mode. Frozen masks retain user-specified nucleotide identities.

When `--ss` is omitted during design, sequence-independent contacts inferred from the coarse-grained target are used only by the thermodynamic screen. `--ss auto` additionally makes those inferred contacts a hard constraint. `--ss none` constructs an all-unpaired, zero-contact DBN and therefore removes all inferred base-pair contacts from both channels. Full CLI semantics and examples are kept in [Inputs and constraints](INPUTS.md#secondary-structure).

After sampling, DS3dRNA estimates a nucleotide distribution at each position and applies dynamic programming to decode a high-probability sequence under chain-aware homopolymer limits. The default maximum run length is five nucleotides.

## DS3dRank

DS3dRank applies the Fine fixed-backbone energy to arbitrary FASTA candidates. It can rank sequences against one scaffold, a directory of independent scaffolds, or a multi-state ensemble. Rank mode does not use thermodynamic reranking.

## Scientific references

The higher-order energy and DS3dRNA workflow should be cited using the entries in [CITATIONS.md](../CITATIONS.md). Thermodynamic parameter and tool references are listed separately so downstream work can attribute the exact components used.
