# Datasets

[← DS3dRNA](../README.md) · [Energy tensors](../Energy/README.md) · [Citations](../CITATIONS.md)

DS3dRNA-associated datasets are hosted separately to keep the source repository focused and within practical clone sizes:

[Open the DS3dRNA dataset folder on Google Drive](https://drive.google.com/drive/folders/1EgsNwqr3uatONFZ98olXPeVfSu-aLiUW)

## Folder contents

The shared folder currently contains the following downloadable archives and index files:

| File | Contents |
| --- | --- |
| `3260_training.zip` | Deduplicated DS3dRNA training structures (3,260 structures). |
| `AlphaFold3_results_All.zip` | Collected AlphaFold 3 structure-validation results used in the accompanying analyses. |
| `CASP17_P20_kissing-multiloop.zip` | CASP17 P20 kissing-multiloop target materials and associated results. |
| `clean_pool_6968.zip` | Cleaned structural candidate pool containing 6,968 representatives. |
| `ssDNA_cif.zip` | Single-stranded DNA structures in CIF/mmCIF format used as the training collection for constructing the DNA energy potential. |
| `T11.zip` | Multi-state benchmark: 11 target clusters comprising 168 conformers in total. |
| `T25.zip` | RNA-only inverse-folding/self-consistency benchmark containing 25 targets. |
| `T83_molecule_report_vs_train.xlsx` | Per-molecule comparison report between the T83 benchmark and the combined training collection. |
| `T83.zip` | Nonredundant benchmark set containing 83 RNA molecules. |
| `Test_pdb_id_all_methods_collection_999.txt` | PDB identifiers in the combined all-method test collection (999 identifiers). |
| `Train_pdb_id_all_methods_collection_6414.txt` | PDB identifiers in the combined all-method training collection (6,414 identifiers). |

The descriptions above follow the current DS3dRNA method manuscript and the folder snapshot documented for this release. Consult metadata packaged inside each download for the definitive target list, provenance, and any archive-specific notes.

`ssDNA_cif.zip` records provenance for the DNA energy tensors distributed under `Energy/DNA/`. It is a training-data archive; the public `DS3dRNA.py` design and ranking interfaces currently accept prepared PDB scaffolds rather than CIF archives.

Dataset users should record the accessed file names and versions in their methods. Cite both DS3dRNA and TriRNASP as described in [CITATIONS.md](../CITATIONS.md), and retain any dataset-specific provenance or license files included with a download.

The external folder's availability and access permissions are managed independently of this repository.
