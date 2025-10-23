#!/usr/bin/env python

"""
Complete Machine Learning Pipeline for Heart Disease Prediction
Implements HRLFM (High-Resolution Logistic-Forest Model)
Targets ≥85% accuracy on cleaned_merged_heart_dataset.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Data processing libraries
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

# Metrics for evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, classification_report, confusion_matrix, roc_curve)

# Balancing technique
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Interpretability tools
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

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import joblib
from pathlib import Path
from datetime import datetime
import time

# Set consistent style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class HeartDiseasePipeline:
    """
    Complete ML Pipeline for Heart Disease Prediction
    Organized into clear steps to enable modular processing and easy explanation.
    """

    def __init__(self, data_path='data/cleaned_merged_heart_dataset.csv', target_accuracy=0.85):
        # Initialize pipeline variables and print start summary
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
        """Step 1: Load dataset and perform initial exploration."""
        print("\n" + "="*80)
        print("STEP 1: DATA LOADING AND EXPLORATION")
        print("="*80)

        # Load dataset
        self.df = pd.read_csv(self.data_path)
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Shape: {self.df.shape}")
        print(f"  Features: {self.df.shape[1] - 1}")
        print(f"  Samples: {self.df.shape[0]}")

        # Show feature types count
        print(f"\nFeature Types:")
        print(f"  Numerical features: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"  Categorical features: {len(self.df.select_dtypes(include=['object']).columns)}")

        # Report missing values if any
        missing = self.df.isnull().sum()
        print(f"\nMissing Values:")
        if missing.sum() == 0:
            print("  ✓ No missing values found")
        else:
            print(missing[missing > 0])

        # Target variable distribution
        print(f"\nTarget Distribution:")
        target_counts = self.df['target'].value_counts()
        print(f"  Class 0 (No Disease): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.df)*100:.2f}%)")
        print(f"  Class 1 (Disease): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.df)*100:.2f}%)")

        # Basic numerical statistics
        print(f"\nBasic Statistics:")
        print(self.df.describe().T[['mean', 'std', 'min', 'max']])

        return self

    def handle_missing_values(self):
        """Step 2: Impute missing values with median for numericals and mode for categoricals."""
        print("\n" + "="*80)
        print("STEP 2: MISSING VALUE HANDLING")
        print("="*80)

        missing_before = self.df.isnull().sum().sum()

        if missing_before > 0:
            print(f"\n⚠ Found {missing_before} missing values")

            # For numerical columns excluding target
            numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col != 'target']

            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"  ✓ Imputed {col} with median: {median_val:.2f}")

            # For categorical columns
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
        """Step 3: Detect and cap outliers using IQR method to reduce noise."""
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
                # Cap outliers instead of removing data points to preserve data
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)

        if outliers_detected:
            print(f"\n⚠ Outliers detected and capped:")
            for col, count in outliers_detected.items():
                print(f"  • {col}: {count} outliers ({count/len(self.df)*100:.2f}%)")
        else:
            print("\n✓ No significant outliers detected")

        return self

    def feature_engineering(self):
        """Step 4: Create new features — polynomial, interaction, domain-specific — to improve model input."""
        print("\n" + "="*80)
        print("STEP 4: FEATURE ENGINEERING")
        print("="*80)

        self.df_engineered = self.df.copy()

        print(f"\nCreating domain-specific features based on data...")

        # Example interaction features, adjust based on actual columns
        if 'age' in self.df.columns:
            if 'chol' in self.df.columns:
                self.df_engineered['age_chol_interaction'] = self.df['age'] * self.df['chol']
                print("  ✓ Created age_chol_interaction")
            if 'trestbps' in self.df.columns:
                self.df_engineered['age_bp_interaction'] = self.df['age'] * self.df['trestbps']
                print("  ✓ Created age_bp_interaction")

        if 'trestbps' in self.df.columns and 'chol' in self.df.columns:
            self.df_engineered['bp_chol_ratio'] = self.df['trestbps'] / (self.df['chol'] + 1)
            print("  ✓ Created bp_chol_ratio")

        if 'thalachh' in self.df.columns and 'age' in self.df.columns:
            self.df_engineered['heart_rate_reserve'] = self.df['thalachh'] - self.df['age']
            print("  ✓ Created heart_rate_reserve")

        if 'oldpeak' in self.df.columns and 'slope' in self.df.columns:
            self.df_engineered['st_depression_severity'] = self.df['oldpeak'] * (self.df['slope'] + 1)
            print("  ✓ Created st_depression_severity")

        if 'cp' in self.df.columns and 'exang' in self.df.columns:
            self.df_engineered['chest_pain_exercise_risk'] = self.df['cp'] * (self.df['exang'] + 1)
            print("  ✓ Created chest_pain_exercise_risk")

        if 'age' in self.df.columns:
            self.df_engineered['age_group'] = pd.cut(
                self.df['age'],
                bins=[0, 40, 55, 70, 120],
                labels=['young', 'middle', 'senior', 'elderly']
            )
            print("  ✓ Created age_group")

        if 'chol' in self.df.columns:
            self.df_engineered['chol_category'] = pd.cut(
                self.df['chol'],
                bins=[0, 200, 240, 600],
                labels=['normal', 'borderline', 'high']
            )
            print("  ✓ Created chol_category")

        # Encode newly created categorical features
        cat_features = self.df_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_features:
            print(f"\nEncoding {len(cat_features)} categorical features...")
            for col in cat_features:
                self.df_engineered[col] = self.df_engineered[col].astype('category').cat.codes
                print(f"  ✓ Encoded {col}")

        # Polynomial features for key columns
        key_features = ['age', 'trestbps', 'chol', 'thalachh', 'oldpeak']
        key_features = [f for f in key_features if f in self.df_engineered.columns]

        if key_features:
            print(f"\nCreating polynomial features (degree 2) for key features...")
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            poly_data = poly.fit_transform(self.df_engineered[key_features])
            poly_feature_names = poly.get_feature_names_out(key_features)

            # Add polynomial features excluding originals
            for i, name in enumerate(poly_feature_names):
                if name not in key_features:
                    self.df_engineered[f'poly_{name}'] = poly_data[:, i]

            print(f"  ✓ Created {len(poly_feature_names) - len(key_features)} polynomial features")

        print(f"\nFeature engineering complete")
        print(f"  Original features: {len(self.df.columns) - 1}")
        print(f"  Engineered features: {len(self.df_engineered.columns) - 1}")
        print(f"  Total new features: {len(self.df_engineered.columns) - len(self.df.columns)}")

        return self

    def prepare_data(self):
        """Step 5: Split data into train/test, scale features, apply SMOTE for balanced dataset."""
        print("\n" + "="*80)
        print("STEP 5: DATA PREPROCESSING AND SPLITTING")
        print("="*80)

        X = self.df_engineered.drop('target', axis=1)
        y = self.df_engineered['target']

        self.feature_names = X.columns.tolist()

        print(f"\nSplitting data (80% train, 20% test) with stratification...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"  ✓ Train samples: {self.X_train.shape[0]}")
        print(f"  ✓ Test samples: {self.X_test.shape[0]}")

        # Scale numeric features robustly — less susceptible to outliers
        print(f"\nScaling features using RobustScaler...")
        self.scaler = RobustScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print("  ✓ Features scaled")

        # SMOTE oversampling to address class imbalance
        print(f"\nBalancing classes with SMOTE...")
        print(f"  Before SMOTE - Class 0: {sum(self.y_train==0)}, Class 1: {sum(self.y_train==1)}")

        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train_scaled, self.y_train)

        print(f"  After SMOTE - Class 0: {sum(self.y_train_balanced==0)}, Class 1: {sum(self.y_train_balanced==1)}")
        print("  ✓ Data balanced")

        return self

    def feature_selection(self):
        """Step 6: Select most relevant features using Random Forest importance and statistical tests."""
        print("\n" + "="*80)
        print("STEP 6: FEATURE SELECTION")
        print("="*80)

        print("\nComputing tree-based feature importance with Random Forest...")
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(self.X_train_balanced, self.y_train_balanced)

        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))

        print("\nSelecting top features using SelectKBest and SelectFromModel...")

        # SelectKBest statistical test (ANOVA F-value)
        k_best = min(40, len(self.feature_names))
        selector_kbest = SelectKBest(f_classif, k=k_best)
        selector_kbest.fit(self.X_train_balanced, self.y_train_balanced)

        # Model-based feature selector threshold set to median importance
        selector_model = SelectFromModel(rf_selector, prefit=True, threshold='median')

        kbest_features = [self.feature_names[i] for i in selector_kbest.get_support(indices=True)]
        model_features = [self.feature_names[i] for i in selector_model.get_support(indices=True)]

        # Final feature set is the union of both
        self.selected_features = list(set(kbest_features) | set(model_features))

        print(f"  ✓ SelectKBest selected: {len(kbest_features)} features")
        print(f"  ✓ SelectFromModel selected: {len(model_features)} features")
        print(f"  ✓ Final selected (union): {len(self.selected_features)} features")

        selected_indices = [i for i, f in enumerate(self.feature_names) if f in self.selected_features]
        self.X_train_selected = self.X_train_balanced[:, selected_indices]
        self.X_test_selected = self.X_test_scaled[:, selected_indices]

        self._save_feature_importance_plot(feature_importance)

        return self

    def train_baseline_models(self):
        """Step 7: Train baseline ML models with hyperparameter tuning for comparison."""
        print("\n" + "="*80)
        print("STEP 7: BASELINE MODEL TRAINING")
        print("="*80)

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

        X_train = self.X_train_selected
        X_test = self.X_test_selected
        y_train = self.y_train_balanced
        y_test = self.y_test

        for model_name, config in model_configs.items():
            print(f"\nTraining {model_name}...")
            start_time = time.time()

            # RandomizedSearchCV for hyperparameter tuning
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=20,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_
            self.baseline_models[model_name] = best_model

            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]

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

            print(f"  Training completed in {elapsed_time:.2f}s")
            print(f"  Best params: {search.best_params_}")
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test ROC-AUC: {roc_auc:.4f}")

        self._print_model_comparison()

        return self

    # ... Additional methods for ensemble training, HRLFM training, evaluation, and persistence continue ...

    def run(self):
        """Run the full pipeline end-to-end."""
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

            print("\nFINAL SUMMARY:")
            print(f"  Dataset size: {self.df.shape[0]} samples, {self.df.shape[1]-1} features")
            print(f"  Engineered features: {len(self.df_engineered.columns)-1}")
            print(f"  Selected features: {len(self.selected_features)}")
            print(f"  Models trained: {len(self.results)}")
            print(f"  Best model: {max(self.results, key=lambda x: self.results[x]['accuracy'])}")
            print(f"  Best accuracy: {max(r['accuracy'] for r in self.results.values()):.4f}")
            print(f"  Target accuracy (≥85%): {'ACHIEVED' if max(r['accuracy'] for r in self.results.values()) >= 0.85 else 'NOT MET'}")

            print("\n" + "="*80)

        except Exception as e:
            print(f"\nPipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise


if __name__ == '__main__':
    pipeline = HeartDiseasePipeline(
        data_path='data/cleaned_merged_heart_dataset.csv',
        target_accuracy=0.85
    )
    pipeline.run()
