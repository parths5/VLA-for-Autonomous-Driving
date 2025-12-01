#!/bin/bash
# Setup and activate VLAD conda environment

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_PATH="${PROJECT_ROOT}/conda/vlad"

echo "VLAD Conda Environment Setup"
echo "=========================================="

if ! command -v conda &> /dev/null; then
    echo "Loading anaconda module..."
    module load anaconda3/2022.10 || {
        echo "Error: Couldn't load anaconda module. Make sure you're on Bridges-2."
        exit 1
    }
fi

# Check if env already exists
if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists!"
    echo ""
    echo "To activate it, run:"
    echo "  cd $PROJECT_ROOT"
    echo "  module load anaconda3/2022.10"
    echo "  conda activate ./conda/vlad"
    echo ""
    exit 0
fi

# Create the environment
echo "Creating conda environment (this will take a few minutes)..."
mkdir -p "${PROJECT_ROOT}/conda"
conda create --prefix "$ENV_PATH" python=3.8.18 -y

# Activate it
echo ""
echo "Installing PyTorch..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PATH"

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install everything else
echo ""
echo "Installing other dependencies..."
pip install -r "${PROJECT_ROOT}/src/requirements.txt"

echo "=========================================="
echo "Done! Environment is ready."

echo "To activate it next time:"
echo "  cd $PROJECT_ROOT"
echo "  module load anaconda3/2022.10"
echo "  conda activate ./conda/vlad"
echo ""
