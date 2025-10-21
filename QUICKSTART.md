# Quick Start Guide

This guide will help you get the Heart Disease Predictor up and running quickly.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Fastest Method: Automated Setup 🚀

**Single command to setup everything:**

```bash
git clone https://github.com/Sarthaka-Mitra/heart-disease-detector.git
cd heart-disease-detector
bash start_app.sh
```

This will:
1. Create a virtual environment
2. Install all dependencies
3. Train the ML models (takes 5-10 minutes)
4. Launch the Streamlit app

**That's it!** The app will open in your browser at `http://localhost:8501`

---

## Alternative: Manual Installation

If you prefer to do things step-by-step or want more control:

### 1. Clone the Repository

```bash
git clone https://github.com/Sarthaka-Mitra/heart-disease-detector.git
cd heart-disease-detector
```

### 2. Create a Virtual Environment (Recommended)

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Training Models

You have multiple options to train the models:

### Option 1: Python Training Script (Recommended)

Train models using optimized Python scripts:

```bash
# Fastest option with pre-tuned hyperparameters
python train_final_optimized_models.py

# OR comprehensive training with feature engineering
python train_new_models.py
```

### Option 2: Using Helper Scripts

**Linux/Mac:**
```bash
bash train_models.sh
```

This will execute the Jupyter notebook and train models.

### Option 3: Using Jupyter Notebook (For Exploratory Analysis)

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/heart_disease_analysis.ipynb`

3. Run all cells (Cell → Run All)

4. Models will be saved in the `models/` directory

## Running the Application

### Using the Run Script (Recommended)

**Linux/Mac:**
```bash
bash run_app.sh
```

**Windows/Manual:**
```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## Using the Application

1. **Select a Model**: Choose from the sidebar (Logistic Regression, Random Forest, XGBoost, or Best Model)

2. **Enter Patient Data**: Fill in all the clinical parameters in the form

3. **Get Prediction**: Click the "Predict Heart Disease Risk" button

4. **View Results**: See the prediction and probability breakdown

## Troubleshooting

### Models Not Found

If you get an error about missing models:
- Make sure you've run the Jupyter notebook or training script first
- Check that the `models/` directory contains `.pkl` files

### Import Errors

If you get import errors:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Try upgrading pip: `pip install --upgrade pip`

### Port Already in Use

If port 8501 is already in use:
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

## Next Steps

- Explore the Jupyter notebook for detailed model analysis
- Check `data/README.md` for dataset information
- Review `models/README.md` for model usage examples

## Support

For issues or questions, please open an issue on GitHub.
