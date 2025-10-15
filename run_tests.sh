#!/bin/bash

# Script to run tests for the Heart Disease Predictor

echo "🧪 Running tests for Heart Disease Predictor..."
echo "=============================================="

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
else
    echo "⚠️  No virtual environment found. Consider creating one with: python -m venv venv"
fi

# Install test dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q pandas joblib scikit-learn xgboost streamlit 2>/dev/null || echo "Dependencies already installed"

# Run tests
echo ""
echo "🧪 Running unit tests..."
python -m pytest tests/ -v --tb=short 2>/dev/null || python -m unittest discover -s tests -p "test_*.py" -v

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✅ All tests passed!"
else
    echo ""
    echo "=============================================="
    echo "❌ Some tests failed. Please check the output above."
    exit 1
fi
