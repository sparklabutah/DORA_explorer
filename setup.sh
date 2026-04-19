#!/usr/bin/env bash
# One-shot setup (similar to https://github.com/sparklabutah/timewarp setup.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml 2>/dev/null || echo "Environment already exists, skipping creation."

echo "Activating conda environment dora-paper-code..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dora-paper-code

echo "Installing root package (MAB Python dependencies)..."
pip install -e .

echo "Installing TALE-Suite in editable mode..."
pip install -e tale-suite/

echo "Setup complete."
echo "Activate later with: conda activate dora-paper-code"
