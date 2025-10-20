import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib

# Load the data
data = pd.read_csv('heart_disease_data.csv')  # Adjust path as necessary

# Data Preprocessing
# Assuming 'target' is the label column
X = data.drop('target', axis=1)
y = data['target']

# Define categorical and numerical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Create preprocessing pipelines
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        ('xgb', XGBClassifier(eval_metric='logloss'))
    ], voting='soft'))
])

# Hyperparameter tuning
param_grid = {
    'classifier__rf__n_estimators': [50, 100],
    'classifier__rf__max_depth': [None, 10, 20],
    'classifier__xgb__n_estimators': [50, 100],
    'classifier__xgb__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Best Model Accuracy: {accuracy * 100:.2f}%')

# Save the best model if accuracy is greater than 85%
if accuracy > 0.85:
    joblib.dump(best_model, 'best_model.pkl')
    print("Model saved as 'best_model.pkl'")
else:
    print("Model accuracy did not exceed 85%")