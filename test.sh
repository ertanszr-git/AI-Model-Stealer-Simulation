#!/bin/bash

# Quick Test Script - Updated for New Structure
echo "🧪 AI Model Stealer - Quick Test"
echo "================================"

# Virtual environment'ı aktive et
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Güvenli simülasyon testi
echo ""
echo "🔒 Testing safe simulation mode..."
cd simulation/
python simulator.py --config config/simulation.yaml
simulation_result=$?
cd ..

if [ $simulation_result -eq 0 ]; then
    echo "✅ Simulation test passed!"
else
    echo "❌ Simulation test failed!"
fi

echo ""
echo "📊 Test Summary:"
echo "- Safe simulation: $([ $simulation_result -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo ""
echo "🎯 Next steps:"
echo "1. 📚 Read quick start: cat QUICK_START.md"
echo "2. 🔒 Try simulation: cd simulation/ && python simulator.py"
echo "3. ⚠️ For real-world (permission required): cd real_world/"
echo "4. 📖 Read documentation: cat README.md"
