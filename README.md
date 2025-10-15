# Heart Disease Detector 🫀

A comprehensive machine learning project for predicting heart disease using multiple ML algorithms with a professional Streamlit web interface.

## 📋 Overview

This project implements a complete heart disease prediction system using various machine learning models. It includes:
- **Data Analysis**: Jupyter notebook with comprehensive EDA
- **Multiple ML Models**: Logistic Regression, Random Forest, and XGBoost
- **Web Application**: Interactive Streamlit app for real-time predictions
- **Model Comparison**: Performance metrics and visualizations

## 🚀 Features

- ✅ Clean, organized repository structure
- ✅ Comprehensive Jupyter notebook with model training and evaluation
- ✅ Multiple ML models with performance comparison
- ✅ Professional Streamlit UI for deployment
- ✅ Model persistence using joblib
- ✅ Detailed documentation and dataset description

## 📁 Project Structure

```
heart-disease-detector/
│
├── data/                          # Dataset directory
│   ├── heart.csv                  # Heart disease dataset
│   └── README.md                  # Dataset description
│
├── notebooks/                     # Jupyter notebooks
│   └── heart_disease_analysis.ipynb  # Main analysis notebook
│
├── models/                        # Trained models (generated after running notebook)
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── app/                          # Streamlit application
│   └── streamlit_app.py          # Main application file
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
└── README.md                     # This file
```

## 🛠️ Installation

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

### 1. Train Models (Jupyter Notebook)

Run the Jupyter notebook to train models and perform analysis:

```bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

The notebook will:
- Load and explore the dataset
- Perform exploratory data analysis
- Train multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Evaluate and compare model performances
- Save trained models to the `models/` directory

### 2. Run the Streamlit Application

After training the models, launch the web application:

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Make Predictions

1. Select a model from the sidebar
2. Enter patient information in the form
3. Click "Predict Heart Disease Risk"
4. View the prediction results and probability breakdown

## 🤖 Models

The project implements and compares three machine learning models:

1. **Logistic Regression**
   - Simple, interpretable linear model
   - Fast training and prediction
   - Good baseline performance

2. **Random Forest**
   - Ensemble of decision trees
   - Feature importance analysis
   - Robust to overfitting

3. **XGBoost**
   - Gradient boosting algorithm
   - High performance
   - Advanced hyperparameter tuning

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Cross-validation scores

## 📈 Dataset

The dataset contains 303 patient records with 13 clinical features:

- **age**: Age in years
- **sex**: 0 = female, 1 = male
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl
- **restecg**: Resting ECG results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina
- **oldpeak**: ST depression
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels (0-3)
- **thal**: Thalassemia (0-3)
- **target**: 0 = no disease, 1 = disease

See `data/README.md` for detailed feature descriptions.

## 🎯 Model Performance

After training, the notebook displays detailed performance metrics including:
- Confusion matrices
- ROC curves
- Feature importance plots
- Model comparison charts

The best performing model is automatically saved as `best_model.pkl`.

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