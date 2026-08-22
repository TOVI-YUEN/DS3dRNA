#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DS3dRNA installer
#
# What this script does:
#   - Restores model energy tensors from Energy.zip when needed.
#   - Creates / reuses conda environment:
#       ${CONDA_BASE}/envs/DS3dRNA
#   - Installs DS3dRNA-related Python dependencies.
#   - Installs PyTorch CUDA 12.8 wheel.
#   - Writes:
#       ${HOME}/check_DS3dRNA_env.py
#
# Assumptions:
#   - Miniconda/Anaconda already installed.
#   - NVIDIA driver already installed if GPU acceleration is needed.
#   - System CUDA Toolkit is not required for PyTorch pip wheels.
# ============================================================

ENV_NAME="DS3dRNA"
PYTHON_VERSION="3.10"
PYTORCH_CUDA_TAG="cu128"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}"

# Set this to "yes" if you also want torchvision/torchaudio.
INSTALL_TORCH_EXTRA="${INSTALL_TORCH_EXTRA:-no}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENERGY_ARCHIVE="${SCRIPT_DIR}/Energy.zip"
ENERGY_SPECS=(
    "Energy/RNA/Rough.npy:2225792"
    "Energy/RNA/Fine.npy:31270016"
    "Energy/DNA/Rough_D.npy:2225792"
    "Energy/DNA/Fine_D.npy:31270016"
)

energy_tensors_ready() {
    local spec rel_path expected_size actual_size
    for spec in "${ENERGY_SPECS[@]}"; do
        rel_path="${spec%%:*}"
        expected_size="${spec##*:}"
        if [[ ! -f "${SCRIPT_DIR}/${rel_path}" ]]; then
            return 1
        fi
        actual_size="$(stat -c '%s' "${SCRIPT_DIR}/${rel_path}")"
        if [[ "${actual_size}" != "${expected_size}" ]]; then
            return 1
        fi
    done
    return 0
}

echo "============================================================"
echo "[DS3dRNA] DS3dRNA-only installer"
echo "============================================================"
echo "[INFO] Environment name : ${ENV_NAME}"
echo "[INFO] Python version   : ${PYTHON_VERSION}"
echo "[INFO] PyTorch CUDA     : ${PYTORCH_CUDA_TAG}"
echo "[INFO] Torch extra      : ${INSTALL_TORCH_EXTRA}"
echo "============================================================"
echo

# ------------------------------------------------------------
# 0. Restore compressed model energy tensors
# ------------------------------------------------------------
echo "============================================================"
echo "[CHECK] Model energy tensors"
echo "============================================================"

if energy_tensors_ready; then
    echo "[OK] All four model energy tensors are already present."
else
    if [[ ! -f "${ENERGY_ARCHIVE}" ]]; then
        echo "[ERROR] Required model tensors are missing and no archive was found:"
        echo "        ${ENERGY_ARCHIVE}"
        exit 1
    fi
    if ! command -v unzip >/dev/null 2>&1; then
        echo "[ERROR] 'unzip' is required to extract ${ENERGY_ARCHIVE}."
        echo "        Install unzip with your system package manager and rerun this installer."
        exit 1
    fi

    echo "[INFO] Extracting model tensors from: ${ENERGY_ARCHIVE}"
    unzip -tq "${ENERGY_ARCHIVE}" >/dev/null
    unzip -oq "${ENERGY_ARCHIVE}" \
        "Energy/RNA/Rough.npy" \
        "Energy/RNA/Fine.npy" \
        "Energy/DNA/Rough_D.npy" \
        "Energy/DNA/Fine_D.npy" \
        -d "${SCRIPT_DIR}"

    if ! energy_tensors_ready; then
        echo "[ERROR] Energy extraction completed, but one or more tensors have an unexpected size."
        echo "        Delete the incomplete Energy/RNA and Energy/DNA tensor files,"
        echo "        replace Energy.zip with a valid release archive, and rerun the installer."
        exit 1
    fi
    echo "[OK] Model energy tensors extracted and validated."
fi
echo

# ------------------------------------------------------------
# 1. Locate conda
# ------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda was not found in PATH."
    echo "        Please install Miniconda/Anaconda first."
    exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

ENV_PREFIX="${CONDA_BASE}/envs/${ENV_NAME}"

echo "[OK] conda found: $(command -v conda)"
echo "[INFO] conda base : ${CONDA_BASE}"
echo "[INFO] env prefix : ${ENV_PREFIX}"
echo

# ------------------------------------------------------------
# 2. Install mamba if missing
# ------------------------------------------------------------
if ! command -v mamba >/dev/null 2>&1; then
    echo "[INFO] mamba not found. Installing mamba into base..."
    conda install -n base -c conda-forge mamba -y
else
    echo "[OK] mamba found: $(command -v mamba)"
fi
echo

# ------------------------------------------------------------
# 3. GPU / driver information
# ------------------------------------------------------------
echo "============================================================"
echo "[CHECK] NVIDIA driver / GPU"
echo "============================================================"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
else
    echo "[WARN] nvidia-smi not found."
    echo "       CPU mode can still work, but GPU checks will fail."
fi
echo

# ------------------------------------------------------------
# 4. Create environment by absolute prefix
# ------------------------------------------------------------
echo "============================================================"
echo "[STEP] Creating / reusing conda environment"
echo "============================================================"

if [[ -d "${ENV_PREFIX}" ]]; then
    echo "[INFO] Environment already exists:"
    echo "       ${ENV_PREFIX}"
else
    echo "[INFO] Creating environment at:"
    echo "       ${ENV_PREFIX}"

    mamba create -p "${ENV_PREFIX}" \
        --override-channels -c conda-forge \
        python="${PYTHON_VERSION}" pip -y
fi

conda activate "${ENV_PREFIX}"

echo "[OK] Activated environment"
echo "     CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV}"
echo "     CONDA_PREFIX=${CONDA_PREFIX}"
python --version
echo

# ------------------------------------------------------------
# 5. Install DS3dRNA Python packages with pip
# ------------------------------------------------------------
echo "============================================================"
echo "[STEP] Installing DS3dRNA Python packages with pip"
echo "============================================================"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    numpy \
    scipy \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    pyyaml \
    joblib \
    psutil \
    rich \
    biopython \
    networkx \
    openpyxl \
    xlrd \
    einops \
    seaborn

echo

# ------------------------------------------------------------
# 6. Install PyTorch cu128
# ------------------------------------------------------------
echo "============================================================"
echo "[STEP] Installing PyTorch CUDA 12.8 wheel"
echo "============================================================"

if [[ "${INSTALL_TORCH_EXTRA}" == "yes" ]]; then
    python -m pip install --upgrade \
        torch torchvision torchaudio \
        --index-url "${PYTORCH_INDEX_URL}"
else
    python -m pip install --upgrade \
        torch \
        --index-url "${PYTORCH_INDEX_URL}"
fi

echo

# ------------------------------------------------------------
# 7. Write environment checker
# ------------------------------------------------------------
echo "============================================================"
echo "[STEP] Writing environment checker"
echo "============================================================"

cat > "${HOME}/check_DS3dRNA_env.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import importlib


def check_import(label: str, module_name: str | None = None, required: bool = True) -> bool:
    name = module_name or label
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK]   {label:<18} module={name:<18} version={version}")
        return True
    except Exception as e:
        level = "[FAIL]" if required else "[WARN]"
        print(f"{level} {label:<18} module={name:<18} error={e}")
        return not required


def main() -> int:
    ok = True

    print("=" * 76)
    print("[CHECK] DS3dRNA-only environment")
    print("=" * 76)

    print("\n[1] DS3dRNA third-party packages")
    for label, module in [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("sklearn", "sklearn"),
        ("tqdm", "tqdm"),
        ("BioPython", "Bio"),
        ("networkx", "networkx"),
        ("yaml", "yaml"),
        ("joblib", "joblib"),
        ("psutil", "psutil"),
        ("rich", "rich"),
        ("einops", "einops"),
        ("openpyxl", "openpyxl"),
        ("xlrd", "xlrd"),
        ("seaborn", "seaborn"),
    ]:
        ok &= check_import(label, module)

    print("\n[2] PyTorch")
    try:
        import torch
        print("[OK] torch version:", torch.__version__)
        print("[OK] torch CUDA runtime:", torch.version.cuda)
        print("[OK] torch.cuda.is_available():", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("[OK] device count:", torch.cuda.device_count())
            print("[OK] device 0:", torch.cuda.get_device_name(0))
            x = torch.randn(256, 256, device="cuda")
            y = x @ x.T
            print("[OK] CUDA matmul test:", tuple(y.shape))
        else:
            print("[WARN] PyTorch CUDA unavailable. CPU mode can still work.")
    except Exception as e:
        print("[FAIL] torch:", e)
        ok = False

    print("\n" + "=" * 76)
    if ok:
        print("[DONE] Environment check passed.")
        return 0

    print("[DONE] Some checks failed. Review warnings/errors above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
PY

chmod +x "${HOME}/check_DS3dRNA_env.py"

echo "[OK] Checker:"
echo "     python ${HOME}/check_DS3dRNA_env.py"
echo

# ------------------------------------------------------------
# 8. Run checks
# ------------------------------------------------------------
echo "============================================================"
echo "[STEP] Running environment checks"
echo "============================================================"

python "${HOME}/check_DS3dRNA_env.py" || true

echo
echo "============================================================"
echo "[DONE] DS3dRNA-only installation finished"
echo "============================================================"
echo
echo "Activate later:"
echo
echo "    conda activate ${ENV_NAME}"
echo
echo "Useful paths:"
echo
echo "    Env prefix: ${ENV_PREFIX}"
echo
echo "Manual checks:"
echo
echo "    conda activate ${ENV_NAME}"
echo "    python ${HOME}/check_DS3dRNA_env.py"
echo
