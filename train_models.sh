#!/bin/bash

# Script to train models using Jupyter notebook

echo "🚀 Starting model training..."
echo "================================"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
else
    echo "⚠️  No virtual environment found. Consider creating one with: python -m venv venv"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Execute the notebook
echo ""
echo "🧠 Training models (this may take a few minutes)..."
jupyter nbconvert --to notebook --execute notebooks/heart_disease_analysis.ipynb --output heart_disease_analysis_executed.ipynb

# Check if models were created
if [ -f "models/best_model.pkl" ]; then
    echo ""
    echo "================================"
    echo "✅ Model training completed successfully!"
    echo ""
    echo "Generated models:"
    ls -lh models/*.pkl
    echo ""
    echo "You can now run the Streamlit app with:"
    echo "  streamlit run app/streamlit_app.py"
else
    echo ""
    echo "❌ Model training failed. Please check the error messages above."
    exit 1
fi
