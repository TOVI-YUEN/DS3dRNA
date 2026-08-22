# Examples

[← DS3dRNA](../README.md) · [Quick start](../docs/QUICKSTART.md) · [Inputs](../docs/INPUTS.md)

The examples are organized around the public `DS3dRNA.py` entry point.

| Directory | Purpose | Main inputs |
| --- | --- | --- |
| [Design](Design/README.md) | Single-state 3D scaffold design and post-processing | 36-nt PDB and DBN |
| [MultiState_Design](MultiState_Design/README.md) | Joint design across five residue-matched conformations | five PDBs, DBN, frozen mask |
| [Seq_Rank](Seq_Rank/README.md) | DS3dRank evaluation of external FASTA candidates | 720-nt PDB, FASTA, reference CSV |

Run commands from the repository root so that energy and source paths resolve consistently. Bundled outputs are examples of file formats and should not be interpreted as a substitute for a full independent scientific validation.
