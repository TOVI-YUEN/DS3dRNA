# Inputs and constraints

[← DS3dRNA](../README.md) · [Quick start](QUICKSTART.md) · [Outputs](OUTPUTS.md)

## RNA and DNA molecule modes

`--mol` selects the molecular alphabet, model energy tensors, local thermodynamic backend, and allowed constrained base pairs. Accepted values are `RNA` and `DNA`; the default is `RNA`.

| Setting | Energy tensors | Local thermodynamics | Output alphabet | Allowed constrained pairs |
| --- | --- | --- | --- | --- |
| `--mol RNA` | `Energy/RNA/Rough.npy`, `Energy/RNA/Fine.npy` | Turner 2004 RNA parameters | A, U, C, G | AU, UA, CG, GC, UG, GU |
| `--mol DNA` | `Energy/DNA/Rough_D.npy`, `Energy/DNA/Fine_D.npy` | Primer3-derived DNA parameters | A, T, C, G | AT, TA, CG, GC |

Design examples:

```bash
# RNA is the default; the two commands are equivalent
python DS3dRNA.py target_RNA.pdb --batch 10
python DS3dRNA.py target_RNA.pdb --mol RNA --batch 10

# DNA design
python DS3dRNA.py target_DNA.pdb --mol DNA --batch 10
```

The option also applies to DS3dRank:

```bash
python DS3dRNA.py -rank \
  --str target_DNA.pdb \
  --fa DNA_candidates.fasta \
  --mol DNA \
  --rank_out DNA_candidates.rank.csv
```

Internally accepted `U`/`T` representations are normalized as needed, while reported DNA sequences use `T`. Users should nevertheless provide molecule-appropriate residue names and sequence alphabets to make inputs and downstream analyses unambiguous. Energy tensor details are listed in [Energy tensors](../Energy/README.md), and thermodynamic sources are documented in [Method overview](METHOD.md#thermodynamic-screening) and the parameter subdirectories.

## PDB scaffolds

DS3dRNA reads PDB coordinate files and uses the P, C4′, and base-anchor atoms N1 (pyrimidines) or N9 (purines). Residues are indexed by chain, residue number, and insertion code. Alternate locations other than blank or `A` are ignored.

The design length includes canonical nucleic-acid residues and unknown `N` residues. DNA residue names DA, DT, DC, and DG are mapped to the corresponding internal nucleic-acid representation.

For multi-state design, every PDB must represent a residue-matched conformation of the same sequence. Structures with unequal or differently ordered residue sets should be aligned and mapped before use.

## Secondary structure

`--ss` accepts:

- a dot-bracket string;
- a `.dbn`, `.ss`, or text file;
- a directory of files matched to PDB basenames; or
- the literal value `auto` or `none`.

Dot-bracket parsing supports pseudoknot levels `()`, `[]`, `{}`, `<>`, and upper-/lower-case letter pairs. `&` marks a chain break and is excluded from compact residue indexing.

In design mode:

- omitted `--ss`: DS3dRNA infers contacts from the coarse-grained PDB geometry. The resulting DBN is used for local thermodynamic screening, but it is not imposed as a hard base-pair constraint;
- explicit `--ss auto`: the same sequence-independent geometric parser runs on demand, and the inferred DBN is used for both thermodynamic screening and the hard base-pair constraint;
- explicit `--ss none`: DS3dRNA constructs an all-unpaired DBN of the correct residue length, preserving inferred chain breaks. For a nine-residue single chain this is `.........`. The thermodynamic and hard-constraint channels therefore receive zero base pairs, and no model-parsed contact is retained;
- explicit DBN/string/path: the supplied structure is used for both thermodynamic screening and the hard base-pair constraint.

In rank mode, omitted `--ss` disables secondary-structure filtering. `--ss auto` supplies automatically inferred contacts as a hard constraint, `--ss none` supplies an all-unpaired zero-pair DBN, and an explicit DBN/string/path supplies the stated hard constraint. Rank mode never performs thermodynamic reranking.

Examples:

```bash
# Implicit auto-SS for thermodynamic screening only
python DS3dRNA.py target.pdb --batch 10

# Auto-SS used for thermodynamics and as a hard pairing constraint
python DS3dRNA.py target.pdb --ss auto --batch 10

# No contacts at all: an all-dot, zero-pair DBN is generated
python DS3dRNA.py target.pdb --ss none --batch 10
```

Any explicit design-time `--ss` value, including `auto` and `none`, selects the fixed constrained trajectory settings of 10,000 total steps, a 4,000-step tail, and `topE = 10000`. This override is reported by the runner. Omitted `--ss` leaves the user-supplied trajectory options unchanged.

## Frozen positions

`--frz` accepts a direct mask, a file, or a directory of per-target files. Nucleotides `A`, `U`, `C`, `G`, or `T` lock the corresponding position; `-`, `.`, and `_` leave positions designable. Whitespace is ignored, and `&` may be used as a chain separator.

Example:

```text
----AUGC----&GGCU----
```

Frozen constraints apply only to design mode. They are ignored by rank mode.

For batch targets, directory matching checks `<pdb_basename>.frz`, `.frozen`, `.txt`, `.fa`, and `.fasta`. For multi-state jobs, matching uses the multi-state folder name.

## FASTA candidates

Rank mode expects candidate sequences whose compact length matches the scaffold residue count. Standard FASTA headers are recommended. If a file contains one sequence per line without headers, convert it with:

```bash
python Examples/Seq_Rank/fix_fasta_headers.py input.txt candidates.fasta
```

## Target and auxiliary fragment preparation

The public runner accepts one PDB, a directory of independent PDB targets, or a multi-state directory. When using externally assembled motifs or alternative conformations, ensure consistent residue correspondence and physically plausible geometry before design. Structural superposition and assembly are outside DS3dRNA's public CLI.
