#!/bin/bash
# NFL Helmet Impact Detection - Quick Start Script

set -e  # Exit on error

echo "════════════════════════════════════════════════════"
echo "  NFL Helmet Impact Detection - Quick Start"
echo "════════════════════════════════════════════════════"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

echo "✓ Current directory: $(pwd)"
echo ""

# Step 1: Check Python version
echo "1️⃣  Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 not found. Please install Python 3.11+"
    exit 1
fi
echo "✓ Python found"
echo ""

# Step 2: Install dependencies
echo "2️⃣  Installing dependencies..."
echo "   (This may take 10-20 minutes for PyTorch...)"
read -p "   Continue? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e .
    echo "✓ Dependencies installed"
else
    echo "⚠️  Skipped installation. Run 'pip install -e .' manually."
fi
echo ""

# Step 3: Check Kaggle credentials
echo "3️⃣  Checking Kaggle credentials..."
if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; then
    echo "✓ Environment variables set"
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "✓ Found $HOME/.kaggle/kaggle.json"
else
    echo "⚠️  No Kaggle credentials found!"
    echo "   Set them up:"
    echo "   export KAGGLE_USERNAME=your_username"
    echo "   export KAGGLE_KEY=your_api_key"
    echo "   OR create ~/.kaggle/kaggle.json"
fi
echo ""

# Step 4: Quick test
echo "4️⃣  Testing configuration..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from impact_detector.config import load_config
config = load_config('configs/base.yaml')
print('✓ Config loaded successfully!')
print(f'  Data dir: {config.data.raw_dir}')
print(f'  Model: {config.model.architecture}')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ Configuration system working"
else
    echo "⚠️  Config test failed (dependencies may not be installed yet)"
fi
echo ""

echo "════════════════════════════════════════════════════"
echo "  Setup Complete!"
echo "════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Download data:       make download-data"
echo "  2. Build cache (tiny):  make build-cache-tiny"
echo "  3. Train model (tiny):  make train-tiny"
echo "  4. Run UI:              make streamlit"
echo ""
echo "For full documentation: cat README.md"
echo "For setup guide:        cat SETUP.md"
echo "For all commands:       make help"
echo ""
