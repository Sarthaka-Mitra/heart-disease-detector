#!/usr/bin/env python
"""
Train machine learning models on the new heart disease dataset
This script handles data preprocessing, feature engineering, model training, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath='data/heart_disease.csv'):
    """Load and preprocess the heart disease dataset"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['Heart Disease Status'].value_counts()}")
    
    # Handle missing values
    print("\n" + "="*60)
    print("Preprocessing data...")
    print("="*60)
    
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled {col} missing values with median: {median_val:.2f}")
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('Heart Disease Status') if 'Heart Disease Status' in categorical_cols else None
    
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_val}")
    
    return df

def create_features(df):
    """Create additional features through feature engineering"""
    print("\n" + "="*60)
    print("Creating engineered features...")
    print("="*60)
    
    df_eng = df.copy()
    
    # Create risk score features
    if 'Age' in df_eng.columns and 'BMI' in df_eng.columns:
        df_eng['Age_BMI_interaction'] = df_eng['Age'] * df_eng['BMI']
        print("Created Age_BMI_interaction")
    
    if 'Blood Pressure' in df_eng.columns and 'Cholesterol Level' in df_eng.columns:
        df_eng['BP_Chol_ratio'] = df_eng['Blood Pressure'] / (df_eng['Cholesterol Level'] + 1)
        print("Created BP_Chol_ratio")
    
    if 'Triglyceride Level' in df_eng.columns and 'Cholesterol Level' in df_eng.columns:
        df_eng['Trig_Chol_ratio'] = df_eng['Triglyceride Level'] / (df_eng['Cholesterol Level'] + 1)
        print("Created Trig_Chol_ratio")
    
    # Age groups
    if 'Age' in df_eng.columns:
        df_eng['Age_group'] = pd.cut(df_eng['Age'], bins=[0, 35, 50, 65, 100], labels=['Young', 'MiddleAge', 'Senior', 'Elderly'])
        print("Created Age_group")
    
    # BMI categories
    if 'BMI' in df_eng.columns:
        df_eng['BMI_category'] = pd.cut(df_eng['BMI'], bins=[0, 18.5, 25, 30, 50], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        print("Created BMI_category")
    
    return df_eng

def encode_categorical_features(df):
    """Encode categorical features"""
    print("\n" + "="*60)
    print("Encoding categorical features...")
    print("="*60)
    
    # Separate target
    target_col = 'Heart Disease Status'
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Label encode binary/ordinal categorical features
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {list(le.classes_)[:5]}...")  # Show first 5 classes
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"\nEncoded target: {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")
    
    return X, y, label_encoders, le_target

def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation with stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

def main():
    """Main training pipeline"""
    print("="*60)
    print("HEART DISEASE PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data('data/heart_disease.csv')
    
    # Feature engineering
    df = create_features(df)
    
    # Encode features
    X, y, label_encoders, le_target = encode_categorical_features(df)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    print("\n" + "="*60)
    print("Splitting data (75% train, 25% test)...")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Apply SMOTE to balance the training data
    print("\n" + "="*60)
    print("Applying SMOTE to balance training data...")
    print("="*60)
    print(f"Before SMOTE - Class distribution: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")
    print(f"Balanced training set: {X_train_balanced.shape}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Save preprocessing objects
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(le_target, 'models/target_encoder.pkl')
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    joblib.dump(smote, 'models/smote.pkl')
    print("Saved preprocessing objects!")
    
    # Train models
    results = {}
    
    # 1. Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=2000, C=0.5)
    results['Logistic Regression'] = train_and_evaluate_model(
        lr_model, 'Logistic Regression', X_train_scaled, X_test_scaled, y_train_balanced, y_test
    )
    joblib.dump(results['Logistic Regression']['model'], 'models/logistic_regression_model.pkl')
    
    # 2. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=25, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    results['Random Forest'] = train_and_evaluate_model(
        rf_model, 'Random Forest', X_train_balanced, X_test, y_train_balanced, y_test
    )
    joblib.dump(results['Random Forest']['model'], 'models/random_forest_model.pkl')
    
    # 3. XGBoost
    xgb_model = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=9,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        eval_metric='logloss', n_jobs=-1
    )
    results['XGBoost'] = train_and_evaluate_model(
        xgb_model, 'XGBoost', X_train_balanced, X_test, y_train_balanced, y_test
    )
    joblib.dump(results['XGBoost']['model'], 'models/xgboost_model.pkl')
    
    # 4. Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42
    )
    results['Gradient Boosting'] = train_and_evaluate_model(
        gb_model, 'Gradient Boosting', X_train_balanced, X_test, y_train_balanced, y_test
    )
    joblib.dump(results['Gradient Boosting']['model'], 'models/gradient_boosting_model.pkl')
    
    # 5. Neural Network
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
        alpha=0.01, max_iter=500, random_state=42
    )
    results['Neural Network'] = train_and_evaluate_model(
        nn_model, 'Neural Network', X_train_scaled, X_test_scaled, y_train_balanced, y_test
    )
    joblib.dump(results['Neural Network']['model'], 'models/neural_network_model.pkl')
    
    # 6. Ensemble (Voting Classifier)
    print("\n" + "="*60)
    print("Creating Ensemble Model (Voting Classifier)...")
    print("="*60)
    ensemble = VotingClassifier(
        estimators=[
            ('rf', results['Random Forest']['model']),
            ('xgb', results['XGBoost']['model']),
            ('gb', results['Gradient Boosting']['model'])
        ],
        voting='soft'
    )
    results['Ensemble'] = train_and_evaluate_model(
        ensemble, 'Ensemble (Voting)', X_train_balanced, X_test, y_train_balanced, y_test
    )
    joblib.dump(results['Ensemble']['model'], 'models/ensemble_model.pkl')
    
    # Model comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [r['accuracy'] for r in results.values()],
        'Precision': [r['precision'] for r in results.values()],
        'Recall': [r['recall'] for r in results.values()],
        'F1-Score': [r['f1'] for r in results.values()],
        'ROC-AUC': [r['roc_auc'] for r in results.values()],
        'CV ROC-AUC': [r['cv_mean'] for r in results.values()]
    })
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Select best model based on ROC-AUC (better for imbalanced datasets)
    best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
    best_model = results[best_model_name]['model']
    print(f"\nBest Model (by ROC-AUC): {best_model_name}")
    print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"\nBest model saved as 'best_model.pkl'")
    
    # Save model comparison
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print("Model comparison saved to 'models/model_comparison.csv'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"All models saved to 'models/' directory")
    print(f"Best model: {best_model_name}")
    print(f"\nNote: This dataset has weak feature correlations with the target,")
    print(f"which limits model performance. Focus on ROC-AUC for evaluation.")

if __name__ == '__main__':
    main()
