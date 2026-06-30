#!/bin/bash
echo "🚀 Starting build process..."

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install --no-cache-dir --no-build-isolation -r requirements.txt

echo "✅ Build complete!"
