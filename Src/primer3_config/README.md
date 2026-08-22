# Primer3 DNA thermodynamic parameters

The `.dh` and `.ds` files in this directory provide local DNA enthalpy and entropy tables used by DS3dRNA's DNA thermodynamic backend. They originate from Primer3's `primer3_config` parameter set and retain the upstream licensing terms described in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).

Main software citation:

> Andreas Untergasser et al. “Primer3—new capabilities and interfaces.” *Nucleic Acids Research* 40(15), e115 (2012). https://doi.org/10.1093/nar/gks596

Core thermodynamic references represented by the file headers include:

- John SantaLucia Jr. “A unified view of polymer, dumbbell, and oligonucleotide DNA nearest-neighbor thermodynamics.” *PNAS* 95, 1460–1465 (1998). https://doi.org/10.1073/pnas.95.4.1460
- John SantaLucia Jr. and Donald Hicks. “The thermodynamics of DNA structural motifs.” *Annual Review of Biophysics and Biomolecular Structure* 33, 415–440 (2004). https://doi.org/10.1146/annurev.biophys.32.110601.141800
- Salvatore Bommarito, Nicolas Peyret, and John SantaLucia Jr. “Thermodynamic parameters for DNA sequences with dangling ends.” *Nucleic Acids Research* 28, 1929–1934 (2000). https://doi.org/10.1093/nar/28.9.1929
- The Allawi–SantaLucia and Peyret et al. mismatch series cited in the individual `stackmm` file headers.

The parameter files are data tables rather than standalone executables. They should remain byte-for-byte stable unless a deliberate, validated upstream update is performed.
