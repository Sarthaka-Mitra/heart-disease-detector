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

# Load models and preprocessing objects
@st.cache_resource
def load_models():
    """Load pre-trained models and preprocessing objects"""
    model_path = Path(__file__).parent.parent / 'models'
    
    models = {}
    try:
        if (model_path / 'logistic_regression_model.pkl').exists():
            models['Logistic Regression'] = joblib.load(model_path / 'logistic_regression_model.pkl')
        if (model_path / 'random_forest_model.pkl').exists():
            models['Random Forest'] = joblib.load(model_path / 'random_forest_model.pkl')
        if (model_path / 'xgboost_model.pkl').exists():
            models['XGBoost'] = joblib.load(model_path / 'xgboost_model.pkl')
        if (model_path / 'gradient_boosting_model.pkl').exists():
            models['Gradient Boosting'] = joblib.load(model_path / 'gradient_boosting_model.pkl')
        if (model_path / 'neural_network_model.pkl').exists():
            models['Neural Network'] = joblib.load(model_path / 'neural_network_model.pkl')
        if (model_path / 'ensemble_model.pkl').exists():
            models['Ensemble'] = joblib.load(model_path / 'ensemble_model.pkl')
        if (model_path / 'best_model.pkl').exists():
            models['Best Model'] = joblib.load(model_path / 'best_model.pkl')
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None
    
    # Load preprocessing objects
    scaler = None
    label_encoders = None
    feature_names = None
    
    try:
        if (model_path / 'scaler.pkl').exists():
            scaler = joblib.load(model_path / 'scaler.pkl')
        if (model_path / 'label_encoders.pkl').exists():
            label_encoders = joblib.load(model_path / 'label_encoders.pkl')
        if (model_path / 'feature_names.pkl').exists():
            feature_names = joblib.load(model_path / 'feature_names.pkl')
    except Exception as e:
        st.warning(f"Preprocessing objects not fully loaded: {e}")
    
    return models, scaler, label_encoders, feature_names

# Sidebar - Model Selection
st.sidebar.title("⚙️ Configuration")
models, scaler, label_encoders, feature_names = load_models()

if models:
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(models.keys()),
        index=len(models) - 1 if 'Best Model' in models else 0
    )
    model = models[selected_model_name]
else:
    st.error("No models found. Please run the training script first: `python train_new_models.py`")
    st.stop()

# Sidebar - Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info(
    "This application predicts the likelihood of heart disease based on various health parameters. "
    "The models were trained on a comprehensive heart disease dataset with 10,000 patient records."
)

st.sidebar.markdown("### 🎯 Model Info")
st.sidebar.success(f"Currently using: **{selected_model_name}**")

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Model Performance", "ℹ️ About Dataset"])

with tab1:
    st.markdown("### Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
        cholesterol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=400, value=200, step=1)
        exercise = st.selectbox("Exercise Habits", options=["Low", "Medium", "High"])
        smoking = st.selectbox("Smoking", options=["No", "Yes"])
        family_history = st.selectbox("Family Heart Disease", options=["No", "Yes"])
    
    with col2:
        diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        high_bp = st.selectbox("High Blood Pressure", options=["No", "Yes"])
        low_hdl = st.selectbox("Low HDL Cholesterol", options=["No", "Yes"])
        high_ldl = st.selectbox("High LDL Cholesterol", options=["No", "Yes"])
        alcohol = st.selectbox("Alcohol Consumption", options=["None", "Low", "Medium", "High"])
        stress = st.selectbox("Stress Level", options=["Low", "Medium", "High"])
    
    with col3:
        sleep_hours = st.number_input("Sleep Hours per day", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
        sugar_consumption = st.selectbox("Sugar Consumption", options=["Low", "Medium", "High"])
        triglyceride = st.number_input("Triglyceride Level (mg/dl)", min_value=50, max_value=500, value=150, step=1)
        fasting_bs = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=70, max_value=200, value=100, step=1)
        crp_level = st.number_input("CRP Level (mg/L)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        homocysteine = st.number_input("Homocysteine Level (µmol/L)", min_value=4.0, max_value=25.0, value=10.0, step=0.1)
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_dict = {
            'Age': age,
            'Gender': gender,
            'Blood Pressure': blood_pressure,
            'Cholesterol Level': cholesterol,
            'Exercise Habits': exercise,
            'Smoking': smoking,
            'Family Heart Disease': family_history,
            'Diabetes': diabetes,
            'BMI': bmi,
            'High Blood Pressure': high_bp,
            'Low HDL Cholesterol': low_hdl,
            'High LDL Cholesterol': high_ldl,
            'Alcohol Consumption': alcohol,
            'Stress Level': stress,
            'Sleep Hours': sleep_hours,
            'Sugar Consumption': sugar_consumption,
            'Triglyceride Level': triglyceride,
            'Fasting Blood Sugar': fasting_bs,
            'CRP Level': crp_level,
            'Homocysteine Level': homocysteine
        }
        
        # Add engineered features
        input_dict['Age_BMI_interaction'] = age * bmi
        input_dict['BP_Chol_ratio'] = blood_pressure / (cholesterol + 1)
        input_dict['Trig_Chol_ratio'] = triglyceride / (cholesterol + 1)
        
        # Age group
        if age <= 35:
            age_group = 'Young'
        elif age <= 50:
            age_group = 'MiddleAge'
        elif age <= 65:
            age_group = 'Senior'
        else:
            age_group = 'Elderly'
        input_dict['Age_group'] = age_group
        
        # BMI category
        if bmi < 18.5:
            bmi_cat = 'Underweight'
        elif bmi < 25:
            bmi_cat = 'Normal'
        elif bmi < 30:
            bmi_cat = 'Overweight'
        else:
            bmi_cat = 'Obese'
        input_dict['BMI_category'] = bmi_cat
        
        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical features using label encoders
        if label_encoders:
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    try:
                        input_df[col] = le.transform(input_df[col].astype(str))
                    except:
                        # If value not seen during training, use most common class
                        input_df[col] = 0
        
        # Reorder columns to match training order
        if feature_names:
            input_df = input_df[feature_names]
        
        # Make prediction
        try:
            # Check if model needs scaling (Logistic Regression, Neural Network)
            if selected_model_name in ['Logistic Regression', 'Neural Network'] and scaler is not None:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                probability = model.predict_proba(input_scaled)[0]
            else:
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)[0]
            
            # Display results
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 0:
                    st.markdown(
                        '<div class="prediction-box healthy">'
                        '<h2 style="color: #27AE60; text-align: center;">✅ Low Risk</h2>'
                        '<p style="text-align: center; font-size: 1.2rem;">No significant heart disease risk detected</p>'
                        '</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="prediction-box at-risk">'
                        '<h2 style="color: #E74C3C; text-align: center;">⚠️ High Risk</h2>'
                        '<p style="text-align: center; font-size: 1.2rem;">Heart disease risk detected - Please consult a healthcare professional</p>'
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
            st.error("Please ensure all required fields are filled correctly.")

with tab2:
    st.markdown("### 📊 Model Performance Metrics")
    
    # Try to load model comparison
    try:
        model_path = Path(__file__).parent.parent / 'models'
        if (model_path / 'model_comparison.csv').exists():
            comparison_df = pd.read_csv(model_path / 'model_comparison.csv')
            st.markdown("#### Model Comparison")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Visualize metrics
            st.markdown("#### Performance Metrics Visualization")
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
            
            for metric in metrics_to_plot:
                if metric in comparison_df.columns:
                    chart_data = comparison_df.set_index('Model')[metric]
                    st.bar_chart(chart_data)
                    st.markdown(f"**{metric}**")
        else:
            st.info("Model comparison data not available. Run the training script to generate it.")
    except Exception as e:
        st.warning(f"Could not load model comparison: {e}")
    
    st.markdown("""
    The models were evaluated using the following metrics:
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: Proportion of positive predictions that were correct
    - **Recall**: Proportion of actual positives that were identified
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Area under the ROC curve (best metric for imbalanced datasets)
    
    ### Models Available:
    1. **Logistic Regression**: Simple linear model for binary classification
    2. **Random Forest**: Ensemble of decision trees
    3. **XGBoost**: Gradient boosting algorithm
    4. **Gradient Boosting**: Another gradient boosting implementation
    5. **Neural Network**: Multi-layer perceptron
    6. **Ensemble**: Voting classifier combining multiple models
    """)

with tab3:
    st.markdown("### 📋 Dataset Information")
    
    st.markdown("""
    #### Features Description:
    
    **Demographics:**
    - **Age**: Patient's age in years
    - **Gender**: Male or Female
    - **BMI**: Body Mass Index
    
    **Cardiovascular Metrics:**
    - **Blood Pressure**: Measured in mm Hg
    - **Cholesterol Level**: Total cholesterol in mg/dl
    - **Triglyceride Level**: Triglycerides in mg/dl
    - **High Blood Pressure**: Yes/No indicator
    - **Low HDL Cholesterol**: Yes/No indicator
    - **High LDL Cholesterol**: Yes/No indicator
    
    **Lifestyle Factors:**
    - **Exercise Habits**: Low, Medium, or High
    - **Smoking**: Yes/No
    - **Alcohol Consumption**: None, Low, Medium, or High
    - **Sleep Hours**: Hours of sleep per day
    - **Sugar Consumption**: Low, Medium, or High
    - **Stress Level**: Low, Medium, or High
    
    **Medical History:**
    - **Family Heart Disease**: Yes/No for family history
    - **Diabetes**: Yes/No
    - **Fasting Blood Sugar**: In mg/dl
    
    **Biomarkers:**
    - **CRP Level**: C-Reactive Protein level in mg/L (inflammation marker)
    - **Homocysteine Level**: In µmol/L (cardiovascular risk marker)
    
    #### Target:
    - **Heart Disease Status**: Yes or No
    
    #### Dataset Statistics:
    - Total samples: 10,000
    - Features: 20 (plus 5 engineered features)
    - Class distribution: 80% No Disease, 20% Disease
    - Handles imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #7F8C8D;'>"
    "Heart Disease Predictor | Built with Streamlit, Scikit-learn & XGBoost | "
    "For educational purposes only"
    "</p>",
    unsafe_allow_html=True
)
