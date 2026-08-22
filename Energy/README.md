# Energy tensors

[← DS3dRNA](../README.md) · [Method](../docs/METHOD.md) · [Datasets](../Datasets/README.md)

The repository distributes the tabulated three-body statistical potentials in the root-level `Energy.zip` to stay within GitHub's per-file upload limit. Running `bash DS3dRNA_Installer.sh` automatically restores the following runtime layout:

| Molecule | Rough tensor | Fine tensor |
| --- | --- | --- |
| RNA | `RNA/Rough.npy` | `RNA/Fine.npy` |
| DNA | `DNA/Rough_D.npy` | `DNA/Fine_D.npy` |

All four files are one-dimensional `float32` NumPy arrays in the established TriRNASP lookup order. Rough arrays contain 556,416 values; Fine arrays contain 7,817,472 values. The loader reconstructs atom-type and distance-bin indexing at runtime.

The extracted `.npy` files are ignored by Git and should not be uploaded separately. The installer validates the archive before extraction and checks the exact expected byte size of every tensor. Existing valid tensors are left untouched on repeated runs.

These tensors are DS3dRNA model assets and are covered by the repository's DS3dRNA license. Their scientific basis is the three-body framework described in:

> Tongwei Yuan, En Lou, Zouchenyu Zhou, Ya-Lan Tan, and Zhi-Jie Tan. “TriRNASP: A knowledge-based potential with three-body effects for accurate RNA structure evaluation.” *Biophysical Journal* 125(11), 2526–2540 (2026). https://doi.org/10.1016/j.bpj.2026.04.003

The DS3dRNA-specific construction and use of the design potentials should additionally be attributed to the DS3dRNA manuscript listed in [CITATIONS.md](../CITATIONS.md).

The single-stranded DNA structural training collection used for the DNA potential is released separately as `ssDNA_cif.zip`; see [Datasets](../Datasets/README.md#folder-contents).

Do not edit, reshape, or text-convert the extracted arrays. A changed byte order, dtype, length, or element order will invalidate energy lookup. To recover a missing or damaged tensor, remove only the affected extracted `.npy` file and rerun `DS3dRNA_Installer.sh` with a valid `Energy.zip` in the repository root.
