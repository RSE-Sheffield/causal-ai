#!/bin/bash
# Install causal-ai on Bede so it is available via the following:
#   module use /nobackup/projects/bddur53/causal-ai/modulefiles
#   module load causal_ai/0.1.0
#
# Run this script once from a login node:
#   bash scripts/install_bede.sh

set -euo pipefail

PROJECT_DIR="/nobackup/projects/bddur53/causal-ai"
CONDA_BASE="/nobackup/projects/bddur53/cs1fxa/Miniforge"
CONDA_ENV="${CONDA_BASE}/envs/new-ai-4-science"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Installing causal-ai ==="
echo "Source repo:  ${REPO_DIR}"
echo "Install to:   ${PROJECT_DIR}"
echo ""

# 1. Copy project files to shared location
echo "[1/4] Copying project files..."
mkdir -p "${PROJECT_DIR}"
rsync -av --exclude='.git' \
          --exclude='__pycache__' \
          --exclude='.idea' \
          --exclude='test_epic' \
          "${REPO_DIR}/" "${PROJECT_DIR}/"

# 2. Ensure the conda environment exists
echo "[2/4] Checking conda environment..."
if [ ! -d "${CONDA_ENV}" ]; then
    echo "ERROR: Conda environment not found at ${CONDA_ENV}"
    echo "Create it first:  conda env create -f ${REPO_DIR}/environment.yml"
    exit 1
fi

# 3. Install causal-ai into the conda env
echo "[3/4] Installing causal-ai package..."
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate new-ai-4-science
pip install -e "${PROJECT_DIR}" --no-deps
conda deactivate

# 4. Verify modulefile is in place
echo "[4/4] Verifying modulefile..."
MODULEFILE="${PROJECT_DIR}/modulefiles/causal_ai/0.1.0.lua"
if [ -f "${MODULEFILE}" ]; then
    echo "Modulefile OK: ${MODULEFILE}"
else
    echo "ERROR: Modulefile not found at ${MODULEFILE}"
    exit 1
fi

echo ""
echo "=== Installation complete ==="
echo ""
echo "Users can now run:"
echo "  module use ${PROJECT_DIR}/modulefiles"
echo "  module load causal_ai/0.1.0"
echo "  python -m causal_ai --help"
echo ""
echo "To make this permanent, add to ~/.bashrc:"
echo "  module use ${PROJECT_DIR}/modulefiles"
