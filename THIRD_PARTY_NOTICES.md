# Third-party notices

[← DS3dRNA](README.md) · [Citations](CITATIONS.md) · [License files](LICENSES/README.md)

The DS3dRNA license in `LICENSE.txt` applies to DS3dRNA-authored code and model assets. It does not replace the terms of separately authored materials listed below.

## ViennaRNA-format RNA parameter file

Path: `Src/Turner2004_Par/rna_turner2004.par`

This file contains Turner 2004 nearest-neighbor parameters in ViennaRNA parameter-file format. ViennaRNA's published terms permit research, educational, and commercial use and modification subject to attribution and redistribution conditions. The upstream terms are reproduced in [LICENSES/ViennaRNA-parameters.txt](LICENSES/ViennaRNA-parameters.txt). Project documentation: https://viennarna.readthedocs.io/en/latest/license.html

Relevant citations are listed in [Src/Turner2004_Par/README.md](Src/Turner2004_Par/README.md).

## Primer3 thermodynamic parameter files

Paths: `Src/primer3_config/*.dh` and `Src/primer3_config/*.ds`

These parameter tables originate from Primer3's `primer3_config` distribution. Primer3 is distributed under GNU GPL version 2 or, at the user's option, any later version. A copy of GPL version 2 is provided in [LICENSES/GPL-2.0.txt](LICENSES/GPL-2.0.txt). Upstream project: https://github.com/primer3-org/primer3

The individual parameter headers identify the thermodynamic publications from which values originate. Consolidated citations are provided in [Src/primer3_config/README.md](Src/primer3_config/README.md).

## DSSR

`Tool/get_dbn.py` is a DS3dRNA wrapper that can invoke a user-supplied DSSR executable. DSSR itself is not included. Users must register and sign in to the X3DNA Forum before accessing the official download page: http://forum.x3dna.org/downloads/3dna-download/. Licensing information is maintained at https://x3dna.org/.

## GNU Lesser General Public License 2.1

The LGPL 2.1 text that was previously appended to the project license is now provided as a separate file at [LICENSES/LGPL-2.1.txt](LICENSES/LGPL-2.1.txt). This separation keeps the DS3dRNA-authored academic/non-commercial terms distinct from the standard LGPL text; it does not alter the scope or license of any third-party component.

## Python dependencies

PyTorch, NumPy, tqdm, and packages installed by `DS3dRNA_Installer.sh` are obtained from their upstream package channels and are not vendored in this repository. Each remains subject to its own license.

## Scientific attribution

License notices and scientific citations serve different purposes. Users should comply with the terms above and also cite the exact tools and parameter sources used, as summarized in [CITATIONS.md](CITATIONS.md).
