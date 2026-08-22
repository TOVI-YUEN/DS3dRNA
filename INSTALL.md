# Installation

[← DS3dRNA](README.md) · [Quick start](docs/QUICKSTART.md) · [Inputs](docs/INPUTS.md)

DS3dRNA is intended for Linux systems with Python 3.10 or later. A CUDA-capable NVIDIA GPU is strongly recommended for large targets and batch jobs, although the code can execute on CPU.

## Automated Conda installation

The bundled installer expects the standard Linux `unzip` command and an existing Miniconda or Anaconda installation. Before configuring Conda, it restores the four model tensors from the repository's `Energy.zip`, validates their exact sizes, creates or reuses a Conda environment named `DS3dRNA`, installs the scientific Python dependencies, and installs the PyTorch CUDA 12.8 wheel.

```bash
bash DS3dRNA_Installer.sh
conda activate DS3dRNA
python ~/check_DS3dRNA_env.py
python DS3dRNA.py --help
```

The extraction step is idempotent: a repeated installer run skips `Energy.zip` when all four expected tensors are already present with the correct sizes. If a tensor is absent or incomplete, the installer tests the ZIP and restores all four arrays. It extracts only the tensor files and does not overwrite `Energy/README.md`.

The installer then installs `mamba` into the Conda base environment when it is not already available and writes the environment checker to `~/check_DS3dRNA_env.py`.

## Core runtime packages

The DS3dRNA execution path directly requires:

- Python 3.10+
- PyTorch
- NumPy
- tqdm

The installer also provides common analysis and export packages used around the wider workflow. PyTorch, driver, and CUDA compatibility should be checked on the target machine before long runs.

## Environment verification

The following checks should all complete before a production calculation:

```bash
python -m compileall -q DS3dRNA.py Mode Src Tool Examples
python DS3dRNA.py --help
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

If CUDA is unavailable, DS3dRNA reports `Using device: cpu`. Small examples can still run, but large structures and ranking sets may be slow.

## Repository location

Run commands from the repository root. Energy and thermodynamic parameter paths are resolved relative to the repository layout.

`Energy.zip` must remain next to `DS3dRNA_Installer.sh` until the first successful extraction. The installer locates both using its own filesystem path, so the installation command may be launched from another working directory.

## Optional DSSR setup

DS3dRNA contains an internal coarse-grained automatic secondary-structure parser, so DSSR is not required for the default design path. To generate external DBN annotations with DSSR, obtain a licensed executable from the official project and follow [Tool/README.md](Tool/README.md).
