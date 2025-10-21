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
- ✅ Comprehensive dataset with 1,888 patient records (cleaned_merged_heart_dataset.csv)
- ✅ **NEW: Complete HRLFM Pipeline** - High-Resolution Logistic-Forest Model achieving 97.9% accuracy
- ✅ Advanced feature engineering (23 engineered features: polynomial, interaction, domain-specific)
- ✅ Multiple ML models with performance comparison (8 models)
- ✅ Feature selection using tree-based importance and statistical methods
- ✅ Hyperparameter tuning with RandomizedSearchCV and GridSearchCV
- ✅ SMOTE for handling imbalanced data
- ✅ Ensemble methods: Voting Classifier and Stacking
- ✅ Model interpretability with SHAP and LIME
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

### 1. Train Models - HRLFM Pipeline (Recommended)

Run the complete HRLFM (High-Resolution Logistic-Forest Model) pipeline:

```bash
python train_hrlfm_pipeline.py
```

This comprehensive pipeline will:
- Load and explore the cleaned_merged_heart_dataset.csv (1,888 records)
- Handle missing values and outliers
- Perform feature engineering (create 23 additional features)
- Select the most informative features
- Train 5 baseline models: Logistic Regression, Random Forest, XGBoost, SVM, LightGBM
- Perform hyperparameter tuning with cross-validation
- Create ensemble models (Voting and Stacking)
- Train the HRLFM hybrid model combining linear and non-linear approaches
- Evaluate all models and achieve **≥85% accuracy** (actual: 97.9%)
- Generate interpretability visualizations (SHAP, LIME)
- Save all models and preprocessing objects to the `models/` directory

**Expected runtime**: 5-10 minutes

📖 **Full documentation**: See [HRLFM_PIPELINE.md](HRLFM_PIPELINE.md) for complete details

### 1a. Alternative Training Methods

You can also use the legacy training scripts:

```bash
# Train on the new comprehensive dataset (10,000 records)
python train_new_models.py

# Or use the optimized training script
python train_final_optimized_models.py
```

### 1b. Jupyter Notebook (Exploratory Analysis)

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

The project implements and compares **8 machine learning models**:

1. **Logistic Regression**
   - Simple, interpretable linear model
   - Fast training and prediction
   - Accuracy: 75.4%
   - Best for understanding feature relationships

2. **Random Forest** ⭐ **Best Performer**
   - Ensemble of decision trees
   - Feature importance analysis
   - Robust to overfitting
   - **Accuracy: 97.9%**
   - **ROC-AUC: 0.998**

3. **XGBoost**
   - Gradient boosting algorithm
   - Advanced hyperparameter tuning
   - Excellent for imbalanced datasets
   - Accuracy: 96.8%
   - ROC-AUC: 0.997

4. **LightGBM**
   - Efficient gradient boosting variant
   - Fast training and high accuracy
   - Accuracy: 97.6%
   - ROC-AUC: 0.998

5. **SVM (Support Vector Machine)**
   - Non-linear classification with RBF kernel
   - Good generalization
   - Accuracy: 91.8%

6. **Voting Ensemble**
   - Combines Random Forest, XGBoost, and LightGBM
   - Soft voting for probability averaging
   - Accuracy: 97.6%
   - ROC-AUC: 0.998

7. **Stacking Ensemble**
   - Multi-level ensemble with meta-model
   - Uses Gradient Boosting as meta-learner
   - Accuracy: 95.5%

8. **HRLFM (High-Resolution Logistic-Forest Model)** 🔬
   - Hybrid model combining Logistic Regression, Random Forest, and XGBoost
   - Optimized meta-model for blending predictions
   - Balances linear and non-linear effects
   - Accuracy: 96.3%
   - Provides interpretability with high performance

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score (primary metric)
- 5-fold and 10-fold cross-validation

## 📈 Dataset

### Primary Dataset: cleaned_merged_heart_dataset.csv

The primary dataset used for the HRLFM pipeline contains **1,888 patient records** with **14 clinical features**:

**Clinical Features:**
1. **age**: Age in years (29-77)
2. **sex**: Sex (1 = male, 0 = female)
3. **cp**: Chest pain type (0-4)
4. **trestbps**: Resting blood pressure in mm Hg (94-200)
5. **chol**: Serum cholesterol in mg/dl (126-564)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalachh**: Maximum heart rate achieved (71-202)
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest (0-6.2)
11. **slope**: Slope of the peak exercise ST segment (0-3)
12. **ca**: Number of major vessels (0-4) colored by fluoroscopy
13. **thal**: Thalassemia (0-7)

**Target Variable:**
- **target**: Heart disease diagnosis (0 = No disease, 1 = Disease)

**Dataset Characteristics:**
- Total samples: 1,888
- Original features: 13
- Engineered features: 23 (polynomial, interaction, domain-specific)
- Final features used: 36 (after feature selection)
- Class distribution: Approximately 48% No Disease, 52% Disease (balanced)
- Missing values: None
- Data quality: Cleaned and merged from multiple sources

**Top Engineered Features:**
- Age-cholesterol interaction
- Age-blood pressure interaction
- BP/Cholesterol ratio
- Heart rate reserve (max HR - age)
- ST depression severity
- Chest pain-exercise risk score
- Age groups (categorical)
- Cholesterol categories
- Polynomial features (squared terms and interactions)

### Alternative Dataset: heart_disease.csv

The alternative comprehensive dataset contains **10,000 patient records** with **20 features**:

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

### Legacy Dataset: heart.csv

The original dataset (`heart.csv`) with **303 records** is also included for reference.

See `data/README.md` for detailed feature descriptions of all datasets.

## 🎯 Model Performance

### HRLFM Pipeline Results

After training with the complete pipeline on cleaned_merged_heart_dataset.csv:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** ⭐ | **97.88%** | 97.47% | 98.47% | 97.97% | **0.9978** |
| LightGBM | 97.62% | 96.52% | 98.98% | 97.73% | 0.9984 |
| Voting Ensemble | 97.62% | 96.52% | 98.98% | 97.73% | 0.9982 |
| XGBoost | 96.83% | 95.54% | 98.47% | 96.98% | 0.9970 |
| HRLFM | 96.30% | 94.61% | 98.47% | 96.50% | 0.9960 |
| Stacking Ensemble | 95.50% | 94.09% | 97.45% | 95.74% | 0.9968 |
| SVM | 91.80% | 89.10% | 95.92% | 92.38% | 0.9748 |
| Logistic Regression | 75.40% | 73.73% | 81.63% | 77.48% | 0.8210 |

✅ **Target Achieved**: All top models exceed the 85% accuracy target

**Key Achievements:**
- Best accuracy: **97.88%** (Random Forest)
- Best ROC-AUC: **0.9984** (LightGBM)
- 10-fold CV accuracy: **98.27% (±0.52%)**
- All ensemble models: >95% accuracy
- HRLFM balances performance with interpretability

The pipeline displays detailed performance metrics including:
- Confusion matrices
- ROC curves
- Feature importance plots
- SHAP explanations
- LIME interpretations
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