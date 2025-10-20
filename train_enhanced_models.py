#!/usr/bin/env python
"""
Enhanced training script with comprehensive optimizations to achieve >85% accuracy
This script implements advanced preprocessing, feature engineering, and model optimization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_advanced_preprocess_data(filepath='data/heart_disease.csv'):
    """Load and preprocess with advanced techniques"""
    print("="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"\nDataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Heart Disease Status'].value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df

def create_advanced_features(df):
    """Create advanced features with domain knowledge"""
    print("\n" + "="*80)
    print("CREATING ADVANCED FEATURES")
    print("="*80)
    
    df_eng = df.copy()
    
    # 1. Risk factor combinations
    if all(col in df_eng.columns for col in ['Age', 'BMI', 'Blood Pressure', 'Cholesterol Level']):
        df_eng['Cardiovascular_Risk_Score'] = (
            df_eng['Age'] * 0.3 + 
            df_eng['BMI'] * 0.25 + 
            df_eng['Blood Pressure'] * 0.25 + 
            df_eng['Cholesterol Level'] * 0.2
        )
        print("✓ Created Cardiovascular_Risk_Score")
    
    # 2. Age-related features
    if 'Age' in df_eng.columns:
        df_eng['Age_squared'] = df_eng['Age'] ** 2
        df_eng['Age_cubed'] = df_eng['Age'] ** 3
        df_eng['Is_High_Risk_Age'] = (df_eng['Age'] > 55).astype(int)
        print("✓ Created Age-related features")
    
    # 3. BMI-related features
    if 'BMI' in df_eng.columns:
        df_eng['BMI_squared'] = df_eng['BMI'] ** 2
        df_eng['Is_Obese'] = (df_eng['BMI'] > 30).astype(int)
        df_eng['Is_Overweight'] = ((df_eng['BMI'] >= 25) & (df_eng['BMI'] <= 30)).astype(int)
        print("✓ Created BMI-related features")
    
    # 4. Blood pressure features
    if 'Blood Pressure' in df_eng.columns:
        df_eng['BP_squared'] = df_eng['Blood Pressure'] ** 2
        df_eng['Is_High_BP'] = (df_eng['Blood Pressure'] > 140).astype(int)
        print("✓ Created Blood Pressure features")
    
    # 5. Cholesterol features
    if 'Cholesterol Level' in df_eng.columns and 'Triglyceride Level' in df_eng.columns:
        df_eng['Cholesterol_squared'] = df_eng['Cholesterol Level'] ** 2
        df_eng['Is_High_Cholesterol'] = (df_eng['Cholesterol Level'] > 240).astype(int)
        df_eng['Lipid_Risk_Score'] = df_eng['Cholesterol Level'] + df_eng['Triglyceride Level'] * 0.5
        print("✓ Created Cholesterol features")
    
    # 6. Interaction features
    if all(col in df_eng.columns for col in ['Age', 'BMI']):
        df_eng['Age_BMI_interaction'] = df_eng['Age'] * df_eng['BMI']
        print("✓ Created Age_BMI_interaction")
    
    if all(col in df_eng.columns for col in ['Age', 'Blood Pressure']):
        df_eng['Age_BP_interaction'] = df_eng['Age'] * df_eng['Blood Pressure']
        print("✓ Created Age_BP_interaction")
    
    if all(col in df_eng.columns for col in ['BMI', 'Blood Pressure']):
        df_eng['BMI_BP_interaction'] = df_eng['BMI'] * df_eng['Blood Pressure']
        print("✓ Created BMI_BP_interaction")
    
    if all(col in df_eng.columns for col in ['Blood Pressure', 'Cholesterol Level']):
        df_eng['BP_Chol_ratio'] = df_eng['Blood Pressure'] / (df_eng['Cholesterol Level'] + 1)
        df_eng['BP_Chol_product'] = df_eng['Blood Pressure'] * df_eng['Cholesterol Level']
        print("✓ Created BP_Cholesterol features")
    
    # 7. Lifestyle risk score
    lifestyle_cols = ['Smoking', 'Exercise Habits', 'Alcohol Consumption', 'Stress Level']
    if all(col in df_eng.columns for col in lifestyle_cols):
        # Encode temporarily for score
        lifestyle_score = 0
        if 'Smoking' in df_eng.columns:
            lifestyle_score += (df_eng['Smoking'] == 'Yes').astype(int) * 2
        if 'Exercise Habits' in df_eng.columns:
            exercise_map = {'Low': 2, 'Medium': 1, 'High': 0}
            lifestyle_score += df_eng['Exercise Habits'].map(exercise_map).fillna(1)
        df_eng['Lifestyle_Risk_Score'] = lifestyle_score
        print("✓ Created Lifestyle_Risk_Score")
    
    # 8. Medical history score
    medical_cols = ['Family Heart Disease', 'Diabetes', 'High Blood Pressure']
    if all(col in df_eng.columns for col in medical_cols):
        medical_score = 0
        for col in medical_cols:
            medical_score += (df_eng[col] == 'Yes').astype(int)
        df_eng['Medical_History_Score'] = medical_score
        print("✓ Created Medical_History_Score")
    
    # 9. Biomarker risk
    if all(col in df_eng.columns for col in ['CRP Level', 'Homocysteine Level']):
        df_eng['Biomarker_Risk'] = df_eng['CRP Level'] * 0.5 + df_eng['Homocysteine Level'] * 0.5
        df_eng['High_CRP'] = (df_eng['CRP Level'] > 10).astype(int)
        print("✓ Created Biomarker features")
    
    # 10. Sleep and stress interaction
    if all(col in df_eng.columns for col in ['Sleep Hours', 'Stress Level']):
        sleep_map = df_eng['Sleep Hours'].copy()
        df_eng['Sleep_Deficiency'] = (sleep_map < 6).astype(int)
        df_eng['Excessive_Sleep'] = (sleep_map > 9).astype(int)
        print("✓ Created Sleep features")
    
    print(f"\nTotal features after engineering: {df_eng.shape[1]}")
    return df_eng

def encode_and_impute(df):
    """Encode categorical features and handle missing values with KNN imputation"""
    print("\n" + "="*80)
    print("ENCODING AND IMPUTATION")
    print("="*80)
    
    # Separate target
    target_col = 'Heart Disease Status'
    y = df[target_col].copy()
    X = df.drop(columns=[target_col])
    
    # Label encode categorical features
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values by treating them as a separate category
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"✓ Encoded {col}")
    
    # Use KNN Imputer for missing values (better than mean/median)
    print("\nApplying KNN Imputation...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    print(f"✓ Imputed missing values using KNN (remaining missing: {X_imputed.isnull().sum().sum()})")
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    print(f"✓ Encoded target: {list(le_target.classes_)} -> {list(range(len(le_target.classes_)))}")
    
    return X_imputed, y, label_encoders, le_target, imputer

def select_best_features(X, y, k=30):
    """Select top k features using mutual information"""
    print("\n" + "="*80)
    print(f"FEATURE SELECTION (top {k} features)")
    print("="*80)
    
    # Use mutual information for feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"✓ Selected {len(selected_features)} features out of {X.shape[1]}")
    print(f"Selected features: {selected_features[:10]}..." if len(selected_features) > 10 else f"Selected features: {selected_features}")
    
    return pd.DataFrame(X_selected, columns=selected_features), selector, selected_features

def train_optimized_model(model, model_name, param_grid, X_train, X_test, y_train, y_test, use_calibration=False):
    """Train model with GridSearchCV and extensive hyperparameter tuning"""
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name}")
    print(f"{'='*80}")
    
    # GridSearchCV with stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"Performing GridSearchCV with {len(param_grid)} parameter combinations...")
    grid_search = GridSearchCV(
        model, param_grid, cv=skf, scoring='accuracy', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
    
    # Apply calibration if requested
    if use_calibration:
        print("Applying probability calibration...")
        best_model = CalibratedClassifierCV(best_model, cv=3, method='sigmoid')
        best_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))
    
    return {
        'model': best_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'best_params': grid_search.best_params_
    }

def main():
    """Main training pipeline with comprehensive optimizations"""
    print("\n" + "="*80)
    print("ENHANCED HEART DISEASE PREDICTION - MODEL TRAINING")
    print("Target: Achieve >85% Accuracy for All Models")
    print("="*80)
    
    # Load data
    df = load_and_advanced_preprocess_data('data/heart_disease.csv')
    
    # Advanced feature engineering
    df = create_advanced_features(df)
    
    # Encode and impute
    X, y, label_encoders, le_target, imputer = encode_and_impute(df)
    
    # Feature selection
    X, feature_selector, selected_features = select_best_features(X, y, k=40)
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Split data
    print("\n" + "="*80)
    print("DATA SPLITTING")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape} (80%)")
    print(f"Test set: {X_test.shape} (20%)")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Apply SMOTE
    print("\n" + "="*80)
    print("APPLYING SMOTE")
    print("="*80)
    print(f"Before SMOTE: {np.bincount(y_train)}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {np.bincount(y_train_balanced)}")
    print(f"Balanced training set: {X_train_balanced.shape}")
    
    # Scale features
    print("\nScaling features with StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled")
    
    # Save preprocessing objects
    print("\nSaving preprocessing objects...")
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(le_target, 'models/target_encoder.pkl')
    joblib.dump(selected_features, 'models/feature_names.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(feature_selector, 'models/feature_selector.pkl')
    print("✓ Saved preprocessing objects")
    
    # Train models with extensive hyperparameter tuning
    results = {}
    
    # 1. Logistic Regression
    lr_params = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000],
        'class_weight': ['balanced']
    }
    lr_model = LogisticRegression(random_state=42)
    results['Logistic Regression'] = train_optimized_model(
        lr_model, 'Logistic Regression', lr_params,
        X_train_scaled, X_test_scaled, y_train_balanced, y_test
    )
    joblib.dump(results['Logistic Regression']['model'], 'models/logistic_regression_model.pkl')
    
    # 2. Random Forest
    rf_params = {
        'n_estimators': [300, 500],
        'max_depth': [20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt'],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    results['Random Forest'] = train_optimized_model(
        rf_model, 'Random Forest', rf_params,
        X_train_balanced, X_test, y_train_balanced, y_test
    )
    joblib.dump(results['Random Forest']['model'], 'models/random_forest_model.pkl')
    
    # 3. XGBoost
    xgb_params = {
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'scale_pos_weight': [1, 2]
    }
    xgb_model = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    results['XGBoost'] = train_optimized_model(
        xgb_model, 'XGBoost', xgb_params,
        X_train_balanced, X_test, y_train_balanced, y_test
    )
    joblib.dump(results['XGBoost']['model'], 'models/xgboost_model.pkl')
    
    # 4. Gradient Boosting
    gb_params = {
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'max_depth': [5, 7],
        'subsample': [0.8],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    gb_model = GradientBoostingClassifier(random_state=42)
    results['Gradient Boosting'] = train_optimized_model(
        gb_model, 'Gradient Boosting', gb_params,
        X_train_balanced, X_test, y_train_balanced, y_test
    )
    joblib.dump(results['Gradient Boosting']['model'], 'models/gradient_boosting_model.pkl')
    
    # 5. Neural Network
    nn_params = {
        'hidden_layer_sizes': [(100, 50), (150, 75)],
        'activation': ['relu'],
        'alpha': [0.001, 0.01],
        'learning_rate': ['adaptive'],
        'max_iter': [1000]
    }
    nn_model = MLPClassifier(random_state=42, early_stopping=True)
    results['Neural Network'] = train_optimized_model(
        nn_model, 'Neural Network', nn_params,
        X_train_scaled, X_test_scaled, y_train_balanced, y_test
    )
    joblib.dump(results['Neural Network']['model'], 'models/neural_network_model.pkl')
    
    # 6. SVM (Support Vector Machine)
    svm_params = {
        'C': [1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale'],
        'class_weight': ['balanced']
    }
    svm_model = SVC(random_state=42, probability=True)
    results['SVM'] = train_optimized_model(
        svm_model, 'SVM', svm_params,
        X_train_scaled, X_test_scaled, y_train_balanced, y_test,
        use_calibration=True
    )
    joblib.dump(results['SVM']['model'], 'models/svm_model.pkl')
    
    # 7. Stacking Ensemble (advanced ensemble method)
    print(f"\n{'='*80}")
    print("CREATING STACKING ENSEMBLE")
    print(f"{'='*80}")
    
    # Use best models as base estimators
    estimators = [
        ('rf', results['Random Forest']['model']),
        ('xgb', results['XGBoost']['model']),
        ('gb', results['Gradient Boosting']['model']),
        ('svm', results['SVM']['model'])
    ]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
        n_jobs=-1
    )
    
    print("Training stacking ensemble...")
    stacking_model.fit(X_train_balanced, y_train_balanced)
    
    y_pred = stacking_model.predict(X_test)
    y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nStacking Ensemble Results:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    results['Stacking Ensemble'] = {
        'model': stacking_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'best_params': 'N/A (stacking)'
    }
    joblib.dump(stacking_model, 'models/ensemble_model.pkl')
    
    # Model Comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{r['accuracy']:.4f}" for r in results.values()],
        'Accuracy %': [f"{r['accuracy']*100:.2f}%" for r in results.values()],
        'Precision': [f"{r['precision']:.4f}" for r in results.values()],
        'Recall': [f"{r['recall']:.4f}" for r in results.values()],
        'F1-Score': [f"{r['f1']:.4f}" for r in results.values()],
        'ROC-AUC': [f"{r['roc_auc']:.4f}" for r in results.values()]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Check which models achieved >85% accuracy
    print("\n" + "="*80)
    print("MODELS ACHIEVING >85% ACCURACY")
    print("="*80)
    
    models_above_85 = []
    for name, result in results.items():
        if result['accuracy'] > 0.85:
            models_above_85.append(name)
            print(f"✓ {name}: {result['accuracy']*100:.2f}%")
    
    if not models_above_85:
        print("⚠ No models achieved >85% accuracy yet.")
        print("This is expected given the weak feature correlations in the dataset.")
    else:
        print(f"\n{len(models_above_85)} model(s) achieved >85% accuracy!")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy*100:.2f}%")
    
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save comparison
    comparison_df.to_csv('models/model_comparison.csv', index=False)
    print("\n✓ Model comparison saved to 'models/model_comparison.csv'")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"All models saved to 'models/' directory")
    print(f"Best model: {best_model_name} ({best_accuracy*100:.2f}%)")

if __name__ == '__main__':
    main()
