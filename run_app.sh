#!/bin/bash

# Script to run the Streamlit application

echo "🚀 Starting Heart Disease Predictor..."
echo "======================================="

# Check if models exist
if [ ! -f "models/best_model.pkl" ]; then
    echo "❌ Models not found!"
    echo ""
    echo "Please train the models first by running:"
    echo "  bash train_models.sh"
    echo ""
    echo "Or run the Jupyter notebook:"
    echo "  jupyter notebook notebooks/heart_disease_analysis.ipynb"
    exit 1
fi

echo "✓ Models found"
echo ""
echo "📊 Launching Streamlit app..."
echo "The app will open in your browser at http://localhost:8501"
echo ""

# Run Streamlit
streamlit run app/streamlit_app.py
