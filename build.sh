#!/bin/bash
echo "🚀 Starting build process..."

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install basic dependencies first
echo "📦 Installing basic dependencies..."
pip install --no-cache-dir -r requirements.txt

# Install PyTorch from official CPU-only index
echo "🔥 Installing PyTorch (CPU version)..."
pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.1.2 \
    torchvision==0.16.2

# Verify installation
echo "✅ Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas version: {pandas.__version__}')"

echo "✅ Build complete!"
