#!/usr/bin/env python
"""
Complete Machine Learning Pipeline for Heart Disease Prediction
This script implements a comprehensive ML pipeline with HRLFM (High-Resolution Logistic-Forest Model)
targeting ≥85% accuracy on the cleaned_merged_heart_dataset.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Data processing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Feature selection
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif

# Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report, confusion_matrix, roc_curve)

# Balancing
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not available. Install with: pip install lime")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import joblib
from pathlib import Path
from datetime import datetime
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class HeartDiseasePipeline:
    """Complete ML Pipeline for Heart Disease Prediction"""
    
    def __init__(self, data_path='data/cleaned_merged_heart_dataset.csv', target_accuracy=0.85):
        """Initialize the pipeline"""
        self.data_path = data_path
        self.target_accuracy = target_accuracy
        self.models = {}
        self.results = {}
        self.best_model = None
        self.hrlfm_model = None
        self.feature_names = None
        self.scaler = None
        self.poly_features = None
        
        print("="*80)
        print(" " * 20 + "HEART DISEASE PREDICTION PIPELINE")
        print("="*80)
        print(f"Target Accuracy: ≥{target_accuracy*100}%")
        print(f"Dataset: {data_path}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
    
    def load_and_explore_data(self):
        """Step 1: Load and explore the dataset"""
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING AND EXPLORATION")
        print("="*80)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Shape: {self.df.shape}")
        print(f"  Features: {self.df.shape[1] - 1}")
        print(f"  Samples: {self.df.shape[0]}")
        
        # Feature types
        print(f"\n📊 Feature Types:")
        print(f"  Numerical features: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"  Categorical features: {len(self.df.select_dtypes(include=['object']).columns)}")
        
        # Missing values
        missing = self.df.isnull().sum()
        print(f"\n🔍 Missing Values:")
        if missing.sum() == 0:
            print("  ✓ No missing values found")
        else:
            print(missing[missing > 0])
        
        # Target distribution
        print(f"\n🎯 Target Distribution:")
        target_counts = self.df['target'].value_counts()
        print(f"  Class 0 (No Disease): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.df)*100:.2f}%)")
        print(f"  Class 1 (Disease): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.df)*100:.2f}%)")
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        print(self.df.describe().T[['mean', 'std', 'min', 'max']])
        
        return self
    
    def handle_missing_values(self):
        """Step 2: Handle missing values with sensible imputation"""
        print("\n" + "="*80)
        print("STEP 2: MISSING VALUE HANDLING")
        print("="*80)
        
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before > 0:
            print(f"\n⚠ Found {missing_before} missing values")
            
            # Impute numerical features with median
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col != 'target']
            
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  ✓ Imputed {col} with median: {median_val:.2f}")
            
            # Impute categorical features with mode
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"  ✓ Imputed {col} with mode: {mode_val}")
            
            missing_after = self.df.isnull().sum().sum()
            print(f"\n✓ Missing values handled: {missing_before} → {missing_after}")
        else:
            print("\n✓ No missing values to handle")
        
        return self
    
    def detect_and_handle_outliers(self):
        """Step 3: Detect and handle outliers using IQR method"""
        print("\n" + "="*80)
        print("STEP 3: OUTLIER DETECTION AND HANDLING")
        print("="*80)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'target']
        
        outliers_detected = {}
        
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if outliers > 0:
                outliers_detected[col] = outliers
                # Cap outliers instead of removing (to preserve data)
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outliers_detected:
            print(f"\n⚠ Outliers detected and capped:")
            for col, count in outliers_detected.items():
                print(f"  • {col}: {count} outliers ({count/len(self.df)*100:.2f}%)")
        else:
            print("\n✓ No significant outliers detected")
        
        return self
    
    def feature_engineering(self):
        """Step 4: Create polynomial, interaction, and domain-specific features"""
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING")
        print("="*80)
        
        self.df_engineered = self.df.copy()
        
        # Domain-specific features based on feature names
        feature_cols = [col for col in self.df.columns if col != 'target']
        
        print(f"\n🔧 Creating domain-specific features...")
        
        # Common heart disease features (adjust based on actual column names)
        # Age-related interactions
        if 'age' in self.df.columns:
            if 'chol' in self.df.columns:
                self.df_engineered['age_chol_interaction'] = self.df['age'] * self.df['chol']
                print("  ✓ Created age_chol_interaction")
            if 'trestbps' in self.df.columns:
                self.df_engineered['age_bp_interaction'] = self.df['age'] * self.df['trestbps']
                print("  ✓ Created age_bp_interaction")
        
        # Blood pressure and cholesterol ratio
        if 'trestbps' in self.df.columns and 'chol' in self.df.columns:
            self.df_engineered['bp_chol_ratio'] = self.df['trestbps'] / (self.df['chol'] + 1)
            print("  ✓ Created bp_chol_ratio")
        
        # Exercise capacity (max heart rate - age)
        if 'thalachh' in self.df.columns and 'age' in self.df.columns:
            self.df_engineered['heart_rate_reserve'] = self.df['thalachh'] - self.df['age']
            print("  ✓ Created heart_rate_reserve")
        
        # ST depression and slope interaction
        if 'oldpeak' in self.df.columns and 'slope' in self.df.columns:
            self.df_engineered['st_depression_severity'] = self.df['oldpeak'] * (self.df['slope'] + 1)
            print("  ✓ Created st_depression_severity")
        
        # Risk score combination
        if 'cp' in self.df.columns and 'exang' in self.df.columns:
            self.df_engineered['chest_pain_exercise_risk'] = self.df['cp'] * (self.df['exang'] + 1)
            print("  ✓ Created chest_pain_exercise_risk")
        
        # Age groups
        if 'age' in self.df.columns:
            self.df_engineered['age_group'] = pd.cut(self.df['age'], 
                                                      bins=[0, 40, 55, 70, 120], 
                                                      labels=['young', 'middle', 'senior', 'elderly'])
            print("  ✓ Created age_group")
        
        # Cholesterol categories
        if 'chol' in self.df.columns:
            self.df_engineered['chol_category'] = pd.cut(self.df['chol'], 
                                                          bins=[0, 200, 240, 600], 
                                                          labels=['normal', 'borderline', 'high'])
            print("  ✓ Created chol_category")
        
        # Encode categorical features created
        cat_features = self.df_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_features:
            print(f"\n📝 Encoding {len(cat_features)} categorical features...")
            for col in cat_features:
                self.df_engineered[col] = self.df_engineered[col].astype('category').cat.codes
                print(f"  ✓ Encoded {col}")
        
        # Polynomial features (degree 2) for key features
        print(f"\n🔢 Creating polynomial features (degree 2)...")
        key_features = ['age', 'trestbps', 'chol', 'thalachh', 'oldpeak']
        key_features = [f for f in key_features if f in self.df_engineered.columns]
        
        if key_features:
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            poly_data = poly.fit_transform(self.df_engineered[key_features])
            poly_feature_names = poly.get_feature_names_out(key_features)
            
            # Add only interaction and squared terms (not original features)
            for i, name in enumerate(poly_feature_names):
                if name not in key_features:  # Skip original features
                    self.df_engineered[f'poly_{name}'] = poly_data[:, i]
            
            print(f"  ✓ Created {len(poly_feature_names) - len(key_features)} polynomial features")
        
        print(f"\n✓ Feature engineering complete")
        print(f"  Original features: {len(self.df.columns) - 1}")
        print(f"  Engineered features: {len(self.df_engineered.columns) - 1}")
        print(f"  Total new features: {len(self.df_engineered.columns) - len(self.df.columns)}")
        
        return self
    
    def prepare_data(self):
        """Step 5: Prepare data for modeling - scale and split"""
        print("\n" + "="*80)
        print("STEP 5: DATA PREPROCESSING AND SPLITTING")
        print("="*80)
        
        # Separate features and target
        X = self.df_engineered.drop('target', axis=1)
        y = self.df_engineered['target']
        
        self.feature_names = X.columns.tolist()
        
        # Split data (stratified)
        print(f"\n📊 Splitting data (80% train, 20% test, stratified)...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  ✓ Train set: {self.X_train.shape[0]} samples")
        print(f"  ✓ Test set: {self.X_test.shape[0]} samples")
        
        # Scale features using RobustScaler (less sensitive to outliers)
        print(f"\n🔄 Scaling features using RobustScaler...")
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"  ✓ Features scaled")
        
        # Handle class imbalance with SMOTE
        print(f"\n⚖️ Handling class imbalance with SMOTE...")
        print(f"  Before SMOTE: Class 0={sum(self.y_train==0)}, Class 1={sum(self.y_train==1)}")
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train_scaled, self.y_train)
        
        print(f"  After SMOTE: Class 0={sum(self.y_train_balanced==0)}, Class 1={sum(self.y_train_balanced==1)}")
        print(f"  ✓ Data balanced")
        
        return self
    
    def feature_selection(self):
        """Step 6: Perform model-based feature selection"""
        print("\n" + "="*80)
        print("STEP 6: FEATURE SELECTION")
        print("="*80)
        
        # Tree-based feature importance
        print(f"\n🌲 Computing tree-based feature importance...")
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(self.X_train_balanced, self.y_train_balanced)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📊 Top 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        # Select top K features using SelectKBest
        print(f"\n🎯 Selecting top features using multiple methods...")
        
        # Method 1: SelectKBest with f_classif
        k_best = min(40, len(self.feature_names))  # Select top 40 or all if less
        selector_kbest = SelectKBest(f_classif, k=k_best)
        selector_kbest.fit(self.X_train_balanced, self.y_train_balanced)
        
        # Method 2: SelectFromModel with Random Forest
        selector_model = SelectFromModel(rf_selector, prefit=True, threshold='median')
        
        # Get selected features
        kbest_features = [self.feature_names[i] for i in selector_kbest.get_support(indices=True)]
        model_features = [self.feature_names[i] for i in selector_model.get_support(indices=True)]
        
        # Use union of both methods
        self.selected_features = list(set(kbest_features) | set(model_features))
        
        print(f"  ✓ SelectKBest selected: {len(kbest_features)} features")
        print(f"  ✓ SelectFromModel selected: {len(model_features)} features")
        print(f"  ✓ Final selection (union): {len(self.selected_features)} features")
        
        # Update datasets with selected features
        selected_indices = [i for i, f in enumerate(self.feature_names) if f in self.selected_features]
        self.X_train_selected = self.X_train_balanced[:, selected_indices]
        self.X_test_selected = self.X_test_scaled[:, selected_indices]
        
        # Save feature importance plot
        self._save_feature_importance_plot(feature_importance)
        
        return self
    
    def train_baseline_models(self):
        """Step 7: Train multiple baseline models with hyperparameter tuning"""
        print("\n" + "="*80)
        print("STEP 7: BASELINE MODEL TRAINING")
        print("="*80)
        
        # Define models with hyperparameters
        model_configs = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'LightGBM': {
                'model': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'num_leaves': [31, 50, 70]
                }
            }
        }
        
        self.baseline_models = {}
        
        # Use selected features
        X_train = self.X_train_selected
        X_test = self.X_test_selected
        y_train = self.y_train_balanced
        y_test = self.y_test
        
        for model_name, config in model_configs.items():
            print(f"\n🤖 Training {model_name}...")
            start_time = time.time()
            
            # RandomizedSearchCV for faster hyperparameter tuning
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=20,  # Number of random combinations to try
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            
            # Best model
            best_model = search.best_estimator_
            self.baseline_models[model_name] = best_model
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
            
            # Test predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            elapsed_time = time.time() - start_time
            
            self.results[model_name] = {
                'model': best_model,
                'best_params': search.best_params_,
                'cv_roc_auc': cv_scores.mean(),
                'cv_roc_auc_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'training_time': elapsed_time
            }
            
            print(f"  ✓ Training completed in {elapsed_time:.2f}s")
            print(f"  • Best params: {search.best_params_}")
            print(f"  • CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  • Test Accuracy: {accuracy:.4f}")
            print(f"  • Test ROC-AUC: {roc_auc:.4f}")
        
        self._print_model_comparison()
        
        return self
    
    def train_ensemble_models(self):
        """Step 8: Train ensemble models (Voting and Stacking)"""
        print("\n" + "="*80)
        print("STEP 8: ENSEMBLE MODELING")
        print("="*80)
        
        X_train = self.X_train_selected
        X_test = self.X_test_selected
        y_train = self.y_train_balanced
        y_test = self.y_test
        
        # Voting Classifier (Soft voting)
        print(f"\n🗳️ Training Voting Classifier (Soft Voting)...")
        start_time = time.time()
        
        voting_estimators = [
            ('rf', self.baseline_models['Random Forest']),
            ('xgb', self.baseline_models['XGBoost']),
            ('lgbm', self.baseline_models['LightGBM'])
        ]
        
        voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=-1)
        voting_clf.fit(X_train, y_train)
        
        y_pred = voting_clf.predict(X_test)
        y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]
        
        self.results['Voting Ensemble'] = {
            'model': voting_clf,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'training_time': time.time() - start_time
        }
        
        print(f"  ✓ Voting Ensemble trained in {self.results['Voting Ensemble']['training_time']:.2f}s")
        print(f"  • Accuracy: {self.results['Voting Ensemble']['accuracy']:.4f}")
        print(f"  • ROC-AUC: {self.results['Voting Ensemble']['roc_auc']:.4f}")
        
        # Stacking Classifier
        print(f"\n📚 Training Stacking Classifier...")
        start_time = time.time()
        
        stacking_estimators = [
            ('lr', self.baseline_models['Logistic Regression']),
            ('rf', self.baseline_models['Random Forest']),
            ('xgb', self.baseline_models['XGBoost']),
            ('lgbm', self.baseline_models['LightGBM'])
        ]
        
        # Meta-model: Gradient Boosting
        meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
        
        stacking_clf = StackingClassifier(
            estimators=stacking_estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        stacking_clf.fit(X_train, y_train)
        
        y_pred = stacking_clf.predict(X_test)
        y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
        
        self.results['Stacking Ensemble'] = {
            'model': stacking_clf,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'training_time': time.time() - start_time
        }
        
        print(f"  ✓ Stacking Ensemble trained in {self.results['Stacking Ensemble']['training_time']:.2f}s")
        print(f"  • Accuracy: {self.results['Stacking Ensemble']['accuracy']:.4f}")
        print(f"  • ROC-AUC: {self.results['Stacking Ensemble']['roc_auc']:.4f}")
        
        return self
    
    def train_hrlfm(self):
        """Step 9: Train High-Resolution Logistic-Forest Model"""
        print("\n" + "="*80)
        print("STEP 9: HIGH-RESOLUTION LOGISTIC-FOREST MODEL (HRLFM)")
        print("="*80)
        
        print(f"\n🔬 Training HRLFM - Hybrid Logistic Regression + Random Forest...")
        
        X_train = self.X_train_selected
        X_test = self.X_test_selected
        y_train = self.y_train_balanced
        y_test = self.y_test
        
        # HRLFM: Stacking with Logistic Regression for linear effects and RF for nonlinear
        start_time = time.time()
        
        # Base models
        lr_base = LogisticRegression(C=10, random_state=42, max_iter=1000)
        rf_base = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
        xgb_base = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
        
        # Meta-model: Gradient Boosting for optimal blending
        meta_model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
        
        hrlfm_estimators = [
            ('logistic', lr_base),
            ('random_forest', rf_base),
            ('xgboost', xgb_base)
        ]
        
        hrlfm = StackingClassifier(
            estimators=hrlfm_estimators,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1,
            passthrough=False  # Don't pass original features to meta-model
        )
        
        # Hyperparameter tuning for HRLFM
        print(f"  • Tuning HRLFM hyperparameters...")
        
        param_grid = {
            'final_estimator__n_estimators': [100, 150, 200],
            'final_estimator__max_depth': [3, 5],
            'final_estimator__learning_rate': [0.05, 0.1, 0.15]
        }
        
        hrlfm_search = GridSearchCV(
            hrlfm,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        hrlfm_search.fit(X_train, y_train)
        
        # Best HRLFM model
        self.hrlfm_model = hrlfm_search.best_estimator_
        
        # Cross-validation
        cv_scores = cross_val_score(self.hrlfm_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        
        # Test predictions
        y_pred = self.hrlfm_model.predict(X_test)
        y_pred_proba = self.hrlfm_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        elapsed_time = time.time() - start_time
        
        self.results['HRLFM'] = {
            'model': self.hrlfm_model,
            'best_params': hrlfm_search.best_params_,
            'cv_roc_auc': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'training_time': elapsed_time
        }
        
        print(f"  ✓ HRLFM trained in {elapsed_time:.2f}s")
        print(f"  • Best params: {hrlfm_search.best_params_}")
        print(f"  • CV ROC-AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  • Test Accuracy: {accuracy:.4f}")
        print(f"  • Test Precision: {precision:.4f}")
        print(f"  • Test Recall: {recall:.4f}")
        print(f"  • Test F1-Score: {f1:.4f}")
        print(f"  • Test ROC-AUC: {roc_auc:.4f}")
        
        # Check if target accuracy met
        if accuracy >= self.target_accuracy:
            print(f"\n  ✅ TARGET ACCURACY ACHIEVED: {accuracy:.4f} ≥ {self.target_accuracy}")
        else:
            print(f"\n  ⚠️ Target accuracy not met: {accuracy:.4f} < {self.target_accuracy}")
        
        return self
    
    def final_evaluation(self):
        """Step 10: Final validation and comprehensive evaluation"""
        print("\n" + "="*80)
        print("STEP 10: FINAL VALIDATION AND EVALUATION")
        print("="*80)
        
        # Get best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        self.best_model = self.results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"  • Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        print(f"  • ROC-AUC: {self.results[best_model_name]['roc_auc']:.4f}")
        
        # Detailed evaluation on test set
        X_test = self.X_test_selected
        y_test = self.y_test
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        print(f"\n📊 Detailed Test Set Evaluation:")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, best_model_name)
        
        # Plot ROC curve
        self._plot_roc_curve(y_test, y_pred_proba, best_model_name)
        
        # K-fold cross-validation on full data
        print(f"\n🔄 K-Fold Cross-Validation (k=10) on full dataset:")
        X_full = self.X_train_selected
        y_full = self.y_train_balanced
        
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.best_model, X_full, y_full, cv=kfold, scoring='accuracy', n_jobs=-1)
        
        print(f"  • Mean Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"  • Min Accuracy: {cv_scores.min():.4f}")
        print(f"  • Max Accuracy: {cv_scores.max():.4f}")
        
        return self
    
    def explain_model(self):
        """Step 11: Model interpretability using SHAP and LIME"""
        print("\n" + "="*80)
        print("STEP 11: MODEL INTERPRETABILITY")
        print("="*80)
        
        X_test = self.X_test_selected
        
        # SHAP explanations
        if SHAP_AVAILABLE:
            print(f"\n🔍 Generating SHAP explanations...")
            try:
                # Use TreeExplainer for tree-based models
                if 'Random Forest' in str(type(self.best_model)) or 'XGB' in str(type(self.best_model)):
                    explainer = shap.TreeExplainer(self.best_model)
                    shap_values = explainer.shap_values(X_test[:100])  # Use subset for speed
                    
                    # Handle different SHAP value formats
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification
                    
                    # Summary plot
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(shap_values, X_test[:100], 
                                     feature_names=[self.selected_features[i] for i in range(X_test.shape[1])],
                                     show=False)
                    plt.tight_layout()
                    plt.savefig('models/shap_summary.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ SHAP summary plot saved to models/shap_summary.png")
                else:
                    print(f"  ⚠️ SHAP not supported for this model type")
            except Exception as e:
                print(f"  ⚠️ SHAP explanation failed: {e}")
        else:
            print(f"  ⚠️ SHAP not available")
        
        # LIME explanations
        if LIME_AVAILABLE:
            print(f"\n🔍 Generating LIME explanations...")
            try:
                explainer = LimeTabularExplainer(
                    self.X_train_selected,
                    feature_names=[self.selected_features[i] for i in range(X_test.shape[1])],
                    class_names=['No Disease', 'Disease'],
                    mode='classification',
                    random_state=42
                )
                
                # Explain a few predictions
                idx = 0
                exp = explainer.explain_instance(X_test[idx], self.best_model.predict_proba, num_features=10)
                
                # Save explanation
                exp.save_to_file('models/lime_explanation.html')
                print(f"  ✓ LIME explanation saved to models/lime_explanation.html")
            except Exception as e:
                print(f"  ⚠️ LIME explanation failed: {e}")
        else:
            print(f"  ⚠️ LIME not available")
        
        return self
    
    def save_models(self):
        """Step 12: Save all models and preprocessing objects"""
        print("\n" + "="*80)
        print("STEP 12: MODEL PERSISTENCE")
        print("="*80)
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving models and preprocessing objects...")
        
        # Save all trained models
        for model_name, result in self.results.items():
            model_filename = model_name.lower().replace(' ', '_') + '_model.pkl'
            joblib.dump(result['model'], models_dir / model_filename)
            print(f"  ✓ Saved {model_filename}")
        
        # Save HRLFM as best model if it has highest accuracy
        if 'HRLFM' in self.results and self.results['HRLFM']['accuracy'] >= max(r['accuracy'] for r in self.results.values()):
            joblib.dump(self.hrlfm_model, models_dir / 'best_model.pkl')
            joblib.dump(self.hrlfm_model, models_dir / 'hrlfm_model.pkl')
            print(f"  ✓ Saved best_model.pkl (HRLFM)")
            print(f"  ✓ Saved hrlfm_model.pkl")
        else:
            joblib.dump(self.best_model, models_dir / 'best_model.pkl')
            print(f"  ✓ Saved best_model.pkl")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, models_dir / 'scaler.pkl')
        print(f"  ✓ Saved scaler.pkl")
        
        joblib.dump(self.selected_features, models_dir / 'feature_names.pkl')
        print(f"  ✓ Saved feature_names.pkl")
        
        # Save model comparison
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()]
        }).sort_values('Accuracy', ascending=False)
        
        comparison_df.to_csv(models_dir / 'model_comparison.csv', index=False)
        print(f"  ✓ Saved model_comparison.csv")
        
        print(f"\n✓ All models and objects saved to {models_dir}/")
        
        return self
    
    def _print_model_comparison(self):
        """Print model comparison table"""
        print(f"\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80 + "\n")
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [r['accuracy'] for r in self.results.values()],
            'Precision': [r['precision'] for r in self.results.values()],
            'Recall': [r['recall'] for r in self.results.values()],
            'F1-Score': [r['f1'] for r in self.results.values()],
            'ROC-AUC': [r['roc_auc'] for r in self.results.values()],
            'Time(s)': [r['training_time'] for r in self.results.values()]
        }).sort_values('Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False))
        
    def _save_feature_importance_plot(self, feature_importance):
        """Save feature importance plot"""
        plt.figure(figsize=(12, 10))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Feature importance plot saved to models/feature_importance.png")
    
    def _plot_confusion_matrix(self, cm, model_name):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Confusion matrix saved to models/confusion_matrix.png")
    
    def _plot_roc_curve(self, y_true, y_pred_proba, model_name):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ ROC curve saved to models/roc_curve.png")
    
    def run(self):
        """Run the complete pipeline"""
        start_time = time.time()
        
        try:
            (self
             .load_and_explore_data()
             .handle_missing_values()
             .detect_and_handle_outliers()
             .feature_engineering()
             .prepare_data()
             .feature_selection()
             .train_baseline_models()
             .train_ensemble_models()
             .train_hrlfm()
             .final_evaluation()
             .explain_model()
             .save_models())
            
            total_time = time.time() - start_time
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Total execution time: {total_time/60:.2f} minutes")
            print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Final summary
            print(f"\n📋 FINAL SUMMARY:")
            print(f"  • Dataset: {self.df.shape[0]} samples, {self.df.shape[1]-1} features")
            print(f"  • Engineered features: {len(self.df_engineered.columns)-1}")
            print(f"  • Selected features: {len(self.selected_features)}")
            print(f"  • Models trained: {len(self.results)}")
            print(f"  • Best model: {max(self.results, key=lambda x: self.results[x]['accuracy'])}")
            print(f"  • Best accuracy: {max(r['accuracy'] for r in self.results.values()):.4f}")
            print(f"  • Target accuracy (≥85%): {'✅ ACHIEVED' if max(r['accuracy'] for r in self.results.values()) >= 0.85 else '❌ NOT MET'}")
            
            print("\n" + "="*80)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == '__main__':
    # Run the pipeline
    pipeline = HeartDiseasePipeline(
        data_path='data/cleaned_merged_heart_dataset.csv',
        target_accuracy=0.85
    )
    pipeline.run()
