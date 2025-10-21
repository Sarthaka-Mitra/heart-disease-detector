# Heart Disease Detector 🫀

A comprehensive machine learning project for predicting heart disease using multiple ML algorithms with a professional Streamlit web interface.

## 📋 Overview

This project implements a complete heart disease prediction system using various machine learning models. It includes:
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, Gradient Boosting, Neural Network, and Ensemble
- **Web Application**: Interactive Streamlit app for real-time predictions
- **Advanced Preprocessing**: SMOTE for handling class imbalance, feature engineering
- **Model Comparison**: Performance metrics and visualizations

## 🚀 Features

- ✅ Clean, organized repository structure
- ✅ Comprehensive dataset with 10,000 patient records
- ✅ Advanced feature engineering (interaction terms, categorical groupings)
- ✅ Multiple ML models with performance comparison (6 models)
- ✅ SMOTE for handling imbalanced data
- ✅ Professional Streamlit UI for deployment
- ✅ Model persistence using joblib
- ✅ Detailed documentation and dataset description
- ✅ Unit tests for code validation
- ✅ CI/CD pipeline with GitHub Actions
- ✅ Docker support for containerized deployment
- ✅ Helper scripts for automation

## 📁 Project Structure

```
heart-disease-detector/
│
├── data/                          # Dataset directory
│   ├── heart.csv                  # Original heart disease dataset (303 records)
│   ├── heart_disease.csv          # New comprehensive dataset (10,000 records)
│   └── README.md                  # Dataset description
│
├── notebooks/                     # Jupyter notebooks
│   └── heart_disease_analysis.ipynb  # Original analysis notebook
│
├── models/                        # Trained models (generated after running training)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── neural_network_model.pkl
│   ├── ensemble_model.pkl
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   └── model_comparison.csv
│
├── app/                          # Streamlit application
│   └── streamlit_app.py          # Main application file
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   └── test_app.py               # Application tests
│
├── .github/                      # GitHub Actions workflows
│   └── workflows/
│       └── ci.yml                # CI/CD pipeline
│
├── .streamlit/                   # Streamlit configuration
│   └── config.toml               # App configuration
│
├── requirements.txt              # Python dependencies
├── requirements-dev.txt          # Development dependencies
├── setup.py                      # Package setup
├── start_app.sh                  # 🆕 Complete setup & deployment script (recommended)
├── train_new_models.py           # New training script for updated dataset
├── train_final_optimized_models.py  # Optimized training script
├── Dockerfile                    # Docker image configuration
├── docker-compose.yml            # Docker Compose configuration
├── .gitignore                    # Git ignore file
├── .dockerignore                 # Docker ignore file
├── run_app.sh                    # Application launcher script
├── run_tests.sh                  # Test runner script
├── train_models.sh               # Model training script
└── README.md                     # This file
```

## 🛠️ Installation

### Quick Start (Recommended)

Use the automated setup script that handles everything in one command:

```bash
git clone https://github.com/Sarthaka-Mitra/heart-disease-detector.git
cd heart-disease-detector
bash start_app.sh
```

This script will:
- Create and activate a virtual environment
- Install all dependencies
- Train the machine learning models
- Launch the Streamlit application

**Note:** The script may take 5-10 minutes to complete as it trains multiple ML models.

### Manual Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sarthaka-Mitra/heart-disease-detector.git
   cd heart-disease-detector
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Usage

### 1. Train Models (Recommended Method)

Run the new training script to train models on the comprehensive dataset:

```bash
python train_new_models.py
```

This script will:
- Load and preprocess the heart_disease.csv dataset (10,000 records)
- Perform feature engineering (create interaction terms and categorical groupings)
- Handle missing values
- Apply SMOTE to balance the dataset
- Train multiple ML models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, Neural Network, Ensemble)
- Evaluate and compare model performances
- Save all models and preprocessing objects to the `models/` directory

### 1a. Alternative: Train Models (Jupyter Notebook)

You can also run the original Jupyter notebook for exploratory analysis:

```bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

### 2. Run the Streamlit Application

After training the models, launch the web application:

```bash
streamlit run app/streamlit_app.py
```

Or use the helper script:

```bash
bash run_app.sh
```

The app will open in your browser at `http://localhost:8501`

### 3. Alternative: Use Individual Scripts

For more control, you can use individual helper scripts:

```bash
# Train models only
bash train_models.sh
# OR
python train_final_optimized_models.py
# OR
python train_new_models.py

# Then run the app
bash run_app.sh
```

### 4. Make Predictions

1. Select a model from the sidebar
2. Enter patient information in the form
3. Click "Predict Heart Disease Risk"
4. View the prediction results and probability breakdown

## 🤖 Models

The project implements and compares six machine learning models:

1. **Logistic Regression**
   - Simple, interpretable linear model
   - Fast training and prediction
   - Uses scaled features
   - Best overall performance (ROC-AUC: 0.52, CV ROC-AUC: 0.79)

2. **Random Forest**
   - Ensemble of decision trees
   - Feature importance analysis
   - Robust to overfitting
   - Hyperparameter tuned

3. **XGBoost**
   - Gradient boosting algorithm
   - High cross-validation performance (CV ROC-AUC: 0.91)
   - Advanced hyperparameter tuning
   - Excellent for imbalanced datasets

4. **Gradient Boosting**
   - Another gradient boosting implementation
   - Good baseline performance
   - Complementary to XGBoost

5. **Neural Network**
   - Multi-layer perceptron (100, 50 neurons)
   - Non-linear pattern recognition
   - Uses scaled features

6. **Ensemble (Voting Classifier)**
   - Combines Random Forest, XGBoost, and Gradient Boosting
   - Soft voting for probability averaging
   - Reduces individual model bias

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score (primary metric for imbalanced data)
- Cross-validation scores

## 📈 Dataset

The primary dataset (`heart_disease.csv`) contains 10,000 patient records with 20 clinical features:

**Demographics:**
- **Age**: Age in years
- **Gender**: Male or Female
- **BMI**: Body Mass Index

**Cardiovascular Metrics:**
- **Blood Pressure**: Resting blood pressure (mm Hg)
- **Cholesterol Level**: Total cholesterol (mg/dl)
- **Triglyceride Level**: Triglycerides (mg/dl)
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
- **CRP Level**: C-Reactive Protein level (mg/L) - inflammation marker
- **Homocysteine Level**: In µmol/L - cardiovascular risk marker

**Target Variable:**
- **Heart Disease Status**: Yes (20%) or No (80%)

**Engineered Features:**
- Age_BMI_interaction
- BP_Chol_ratio
- Trig_Chol_ratio
- Age_group (Young, MiddleAge, Senior, Elderly)
- BMI_category (Underweight, Normal, Overweight, Obese)

The original dataset (`heart.csv`) with 303 records is also included for reference.

See `data/README.md` for detailed feature descriptions.

## 🎯 Model Performance

After training, the notebook displays detailed performance metrics including:
- Confusion matrices
- ROC curves
- Feature importance plots
- Model comparison charts

The best performing model is automatically saved as `best_model.pkl`.

## 🧪 Testing

The project includes unit tests to verify functionality:

```bash
# Run all tests
bash run_tests.sh

# Or run tests manually
python -m unittest discover -s tests -v
```

The tests verify:
- Dataset structure and integrity
- Model loading and functionality
- Application imports and configuration

For development testing, install additional dependencies:

```bash
pip install -r requirements-dev.txt
```

## 🐳 Docker Support

Build and run with Docker:

```bash
# Build the image
docker build -t heart-disease-predictor .

# Run the container
docker run -p 8501:8501 heart-disease-predictor

# Or use docker-compose
docker-compose up
```

## 🌐 Deployment

The Streamlit app is ready for deployment on platforms like:
- [Streamlit Cloud](https://streamlit.io/cloud)
- [Heroku](https://www.heroku.com/)
- [AWS](https://aws.amazon.com/)
- [Google Cloud](https://cloud.google.com/)

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set the main file path: `app/streamlit_app.py`
5. Deploy!

## ⚠️ Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ using Python, Scikit-learn, XGBoost, and Streamlit**