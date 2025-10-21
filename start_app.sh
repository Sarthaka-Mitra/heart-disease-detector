#!/bin/bash

# Comprehensive script to setup virtual environment, train models, and start Streamlit app
# This script handles the complete workflow from scratch

# Exit on error, but allow us to handle errors gracefully
set -e

echo "======================================"
echo "🚀 Heart Disease Detector Setup"
echo "======================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Create and activate virtual environment
echo "📦 Step 1: Setting up virtual environment..."
echo "--------------------------------------"

if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Step 2: Install dependencies
echo "📚 Step 2: Installing dependencies..."
echo "--------------------------------------"
echo "Upgrading pip..."
pip install --upgrade pip || echo "⚠️  Warning: Could not upgrade pip, continuing..."
echo "Installing requirements..."
pip install -r requirements.txt || {
    echo "❌ Error: Failed to install dependencies"
    echo "Please ensure you have a working internet connection"
    exit 1
}

echo "✓ Dependencies installed"
echo ""

# Step 3: Train models
echo "🧠 Step 3: Training machine learning models..."
echo "--------------------------------------"
echo "This may take several minutes..."
echo ""

# Check if models directory exists
if [ ! -d "models" ]; then
    mkdir models
    echo "Created models directory"
fi

# Run the final optimized training script (fastest with pre-tuned parameters)
if python train_final_optimized_models.py; then
    echo ""
    echo "✓ Model training completed successfully"
else
    echo ""
    echo "⚠️  Model training encountered issues, trying alternative script..."
    if python train_new_models.py; then
        echo "✓ Model training completed with alternative script"
    else
        echo "❌ Error: Model training failed"
        echo "Please check the error messages above"
        exit 1
    fi
fi

echo ""

# Verify models were created
if [ -f "models/best_model.pkl" ]; then
    echo "✓ Models successfully trained and saved"
    echo ""
    echo "Generated model files:"
    ls -lh models/*.pkl 2>/dev/null || echo "  (checking for .pkl files...)"
    echo ""
else
    echo "⚠️  Warning: best_model.pkl not found, but training completed"
    echo "   Available files in models/:"
    ls -lh models/ 2>/dev/null || echo "  (no files found)"
    echo ""
fi

# Step 4: Start Streamlit app
echo "======================================"
echo "🎉 Setup Complete!"
echo "======================================"
echo ""
echo "🌐 Step 4: Starting Streamlit application..."
echo "--------------------------------------"
echo ""
echo "The app will open in your browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Run Streamlit
streamlit run app/streamlit_app.py
