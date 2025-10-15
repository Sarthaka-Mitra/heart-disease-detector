# Trained Models

This directory contains the trained machine learning models for heart disease prediction.

## Models

The following models are generated after running the Jupyter notebook (`notebooks/heart_disease_analysis.ipynb`):

1. **logistic_regression_model.pkl**: Logistic Regression model
2. **random_forest_model.pkl**: Random Forest Classifier model
3. **xgboost_model.pkl**: XGBoost Classifier model
4. **best_model.pkl**: The best performing model based on F1-score
5. **scaler.pkl**: StandardScaler for feature normalization (used with Logistic Regression)

## Usage

These models are automatically loaded by the Streamlit application (`app/streamlit_app.py`).

To use a model directly in Python:

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare input data
input_data = pd.DataFrame({
    'age': [55],
    'sex': [1],
    'cp': [2],
    'trestbps': [140],
    'chol': [250],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [2.0],
    'slope': [1],
    'ca': [0],
    'thal': [2]
})

# For Logistic Regression, scale the data
# For Random Forest and XGBoost, use raw data

# Make prediction
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

## Model Performance

Run the Jupyter notebook to see detailed performance metrics including:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Feature Importance

## Notes

- Models are saved using joblib for efficient serialization
- The scaler is only required for Logistic Regression
- These files are excluded from git via `.gitignore`
- To regenerate models, run the Jupyter notebook
