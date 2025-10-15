import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #E74C3C;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        text-align: center;
        padding-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy {
        background-color: #D5F4E6;
        border: 2px solid #27AE60;
    }
    .at-risk {
        background-color: #FADBD8;
        border: 2px solid #E74C3C;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">❤️ Heart Disease Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced ML-based Heart Disease Detection System</p>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    """Load pre-trained models and scaler"""
    model_path = Path(__file__).parent.parent / 'models'
    
    models = {}
    try:
        if (model_path / 'logistic_regression_model.pkl').exists():
            models['Logistic Regression'] = joblib.load(model_path / 'logistic_regression_model.pkl')
        if (model_path / 'random_forest_model.pkl').exists():
            models['Random Forest'] = joblib.load(model_path / 'random_forest_model.pkl')
        if (model_path / 'xgboost_model.pkl').exists():
            models['XGBoost'] = joblib.load(model_path / 'xgboost_model.pkl')
        if (model_path / 'best_model.pkl').exists():
            models['Best Model'] = joblib.load(model_path / 'best_model.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None
    
    scaler = None
    try:
        if (model_path / 'scaler.pkl').exists():
            scaler = joblib.load(model_path / 'scaler.pkl')
    except Exception as e:
        st.warning(f"Scaler not loaded: {e}")
    
    return models, scaler

# Sidebar - Model Selection
st.sidebar.title("⚙️ Configuration")
models, scaler = load_models()

if models:
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        index=len(models) - 1 if 'Best Model' in models else 0
    )
    model = models[selected_model_name]
else:
    st.error("No models found. Please run the Jupyter notebook first to train the models.")
    st.stop()

# Sidebar - Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    "This application predicts the likelihood of heart disease based on various clinical parameters. "
    "The models were trained on a heart disease dataset with 303 patient records."
)

st.sidebar.markdown("### 🎯 Model Info")
st.sidebar.success(f"Currently using: **{selected_model_name}**")

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Model Performance", "ℹ️ About Dataset"])

with tab1:
    st.markdown("### Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50, step=1)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])
        cp = st.selectbox("Chest Pain Type", 
                         options=[("Typical Angina", 0), ("Atypical Angina", 1), 
                                 ("Non-anginal Pain", 2), ("Asymptomatic", 3)],
                         format_func=lambda x: x[0])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                   min_value=80, max_value=200, value=120, step=1)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 
                              min_value=100, max_value=600, value=200, step=1)
    
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                          options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
        restecg = st.selectbox("Resting ECG Results", 
                              options=[("Normal", 0), ("ST-T Abnormality", 1), 
                                      ("LV Hypertrophy", 2)],
                              format_func=lambda x: x[0])
        thalach = st.number_input("Maximum Heart Rate Achieved", 
                                 min_value=60, max_value=220, value=150, step=1)
        exang = st.selectbox("Exercise Induced Angina", 
                            options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])
    
    with col3:
        oldpeak = st.number_input("ST Depression (oldpeak)", 
                                 min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                            options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                            format_func=lambda x: x[0])
        ca = st.selectbox("Number of Major Vessels (0-3)", 
                         options=[(0, 0), (1, 1), (2, 2), (3, 3)], 
                         format_func=lambda x: str(x[0]))
        thal = st.selectbox("Thalassemia", 
                           options=[("Normal", 0), ("Fixed Defect", 1), 
                                   ("Reversible Defect", 2), ("Not Described", 3)],
                           format_func=lambda x: x[0])
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex[1]],
            'cp': [cp[1]],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs[1]],
            'restecg': [restecg[1]],
            'thalach': [thalach],
            'exang': [exang[1]],
            'oldpeak': [oldpeak],
            'slope': [slope[1]],
            'ca': [ca[1]],
            'thal': [thal[1]]
        })
        
        # Make prediction
        try:
            # Scale data if using Logistic Regression
            if selected_model_name == 'Logistic Regression' and scaler is not None:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)
                probability = model.predict_proba(input_scaled)[0]
            else:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 0:
                    st.markdown(
                        '<div class="prediction-box healthy">'
                        '<h2 style="color: #27AE60; text-align: center;">✅ Low Risk</h2>'
                        '<p style="text-align: center; font-size: 1.2rem;">No significant heart disease detected</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="prediction-box at-risk">'
                        '<h2 style="color: #E74C3C; text-align: center;">⚠️ High Risk</h2>'
                        '<p style="text-align: center; font-size: 1.2rem;">Heart disease detected - Consult a cardiologist</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
            
            with col2:
                st.markdown("#### Probability Breakdown")
                st.metric("No Disease Probability", f"{probability[0]*100:.2f}%")
                st.metric("Disease Probability", f"{probability[1]*100:.2f}%")
                
                # Probability bar chart
                prob_df = pd.DataFrame({
                    'Category': ['No Disease', 'Disease'],
                    'Probability': [probability[0]*100, probability[1]*100]
                })
                st.bar_chart(prob_df.set_index('Category'))
            
            # Disclaimer
            st.markdown("---")
            st.warning("⚠️ **Disclaimer**: This is a prediction tool and should not replace professional medical advice. "
                      "Please consult with a healthcare professional for proper diagnosis and treatment.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    
    st.info("Run the Jupyter notebook (`notebooks/heart_disease_analysis.ipynb`) to see detailed model performance metrics.")
    
    st.markdown("""
    The models were evaluated using the following metrics:
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: Proportion of positive predictions that were correct
    - **Recall**: Proportion of actual positives that were identified
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Area under the ROC curve
    
    ### Models Trained:
    1. **Logistic Regression**: Simple linear model for binary classification
    2. **Random Forest**: Ensemble of decision trees
    3. **XGBoost**: Gradient boosting algorithm
    """)

with tab3:
    st.markdown("### 📋 Dataset Information")
    
    st.markdown("""
    #### Features Description:
    
    1. **Age**: Patient's age in years
    2. **Sex**: 0 = Female, 1 = Male
    3. **Chest Pain Type (cp)**: 
       - 0: Typical Angina
       - 1: Atypical Angina
       - 2: Non-anginal Pain
       - 3: Asymptomatic
    4. **Resting Blood Pressure (trestbps)**: Measured in mm Hg
    5. **Cholesterol (chol)**: Serum cholesterol in mg/dl
    6. **Fasting Blood Sugar (fbs)**: > 120 mg/dl (1 = true, 0 = false)
    7. **Resting ECG (restecg)**:
       - 0: Normal
       - 1: ST-T wave abnormality
       - 2: Left ventricular hypertrophy
    8. **Maximum Heart Rate (thalach)**: Maximum heart rate achieved
    9. **Exercise Induced Angina (exang)**: 1 = Yes, 0 = No
    10. **ST Depression (oldpeak)**: Induced by exercise relative to rest
    11. **Slope**: Slope of the peak exercise ST segment
        - 0: Upsloping
        - 1: Flat
        - 2: Downsloping
    12. **Number of Major Vessels (ca)**: Colored by fluoroscopy (0-3)
    13. **Thalassemia (thal)**:
        - 0: Normal
        - 1: Fixed defect
        - 2: Reversible defect
        - 3: Not described
    
    #### Target:
    - **0**: No heart disease
    - **1**: Heart disease present
    
    #### Dataset Statistics:
    - Total samples: 303
    - Features: 13
    - Binary classification problem
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7F8C8D;'>"
    "Heart Disease Predictor | Built with Streamlit & Scikit-learn | "
    "For educational purposes only"
    "</p>",
    unsafe_allow_html=True
)
