#!/bin/bash

# AI Model Extraction Simulator Setup Script
echo "🚀 AI Model Extraction Simulator Setup"
echo "=" $(printf "%0.s=" {1..40})

# Python versiyonu kontrolü
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "Found: $python_version"

# Virtual environment oluştur
echo "📦 Creating virtual environment..."
if command -v python3 &> /dev/null; then
    python3 -m venv venv
else
    python -m venv venv
fi

# Virtual environment'ı aktive et
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Pip'i güncelle
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Requirements'ları yükle
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Veri dizinlerini oluştur
echo "📁 Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p experiments
mkdir -p logs

# Test çalıştır
echo "🧪 Running quick test..."
python -c "
import torch
import numpy as np
print('✅ PyTorch version:', torch.__version__)
print('✅ NumPy version:', np.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ Setup completed successfully!')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📚 Choose your learning path:"
echo ""
echo "🔒 For safe learning (RECOMMENDED):"
echo "  cd simulation/"
echo "  python simulator.py"
echo ""
echo "⚠️ For real-world research (PERMISSION REQUIRED):"
echo "  cd real_world/"
echo "  python real_world_attack.py --analyze-only"
echo ""
echo "📖 For quick start guide:"
echo "  cat QUICK_START.md"
echo ""
echo "⚠️  Remember: This tool is for educational purposes only!"
