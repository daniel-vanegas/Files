#!/bin/bash

set -e  # Exit on error

echo "======================================"
echo "Installing PyTorch 2.4.1 (CUDA 12.1)"
echo "======================================"

# Ensure pip is available
if ! command -v pip &> /dev/null; then
    echo "pip not found. Installing python3-pip..."
    sudo apt update
    sudo apt install -y python3-pip
fi

# Install PyTorch from the official CUDA 12.1 wheel
echo "Running: pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --break-system-packages"
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "======================================"
echo "Installation complete!"
echo "======================================"