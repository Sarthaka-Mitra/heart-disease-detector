# Heart Disease Dataset

## Overview
This directory contains two heart disease datasets:
1. **heart.csv**: Original dataset with 303 patient records
2. **heart_disease.csv**: Comprehensive dataset with 10,000 patient records (NEW - Primary dataset)

## Primary Dataset: heart_disease.csv

### Overview
This comprehensive dataset contains 10,000 patient records with 20 clinical features used to predict the presence of heart disease. The dataset includes demographic information, cardiovascular metrics, lifestyle factors, medical history, and biomarkers.

### Features

#### Demographics
1. **Age**: Age in years (18-100)
2. **Gender**: Male or Female
3. **BMI**: Body Mass Index (15-50)

#### Cardiovascular Metrics
4. **Blood Pressure**: Resting blood pressure in mm Hg (80-200)
5. **Cholesterol Level**: Total cholesterol in mg/dl (100-400)
6. **Triglyceride Level**: Triglycerides in mg/dl (50-500)
7. **High Blood Pressure**: Yes/No indicator
8. **Low HDL Cholesterol**: Yes/No indicator (Low "good" cholesterol)
9. **High LDL Cholesterol**: Yes/No indicator (High "bad" cholesterol)

#### Lifestyle Factors
10. **Exercise Habits**: Low, Medium, or High
11. **Smoking**: Yes/No
12. **Alcohol Consumption**: None, Low, Medium, or High
13. **Sleep Hours**: Hours of sleep per day (3-12)
14. **Sugar Consumption**: Low, Medium, or High
15. **Stress Level**: Low, Medium, or High

#### Medical History
16. **Family Heart Disease**: Yes/No for family history of heart disease
17. **Diabetes**: Yes/No
18. **Fasting Blood Sugar**: In mg/dl (70-200)

#### Biomarkers
19. **CRP Level**: C-Reactive Protein level in mg/L (0-20) - A marker of inflammation
20. **Homocysteine Level**: In µmol/L (4-25) - A cardiovascular risk marker

### Target Variable

**Heart Disease Status**: Yes or No
- No: 8,000 samples (80%)
- Yes: 2,000 samples (20%)

### Engineered Features

The training pipeline creates additional features:
- **Age_BMI_interaction**: Age × BMI
- **BP_Chol_ratio**: Blood Pressure / (Cholesterol Level + 1)
- **Trig_Chol_ratio**: Triglyceride Level / (Cholesterol Level + 1)
- **Age_group**: Categorical (Young, MiddleAge, Senior, Elderly)
- **BMI_category**: Categorical (Underweight, Normal, Overweight, Obese)

### Dataset Statistics

- Total samples: 10,000
- Features: 20 (plus 5 engineered features)
- Target classes: 2 (Binary classification)
- Class distribution: 80% No disease, 20% Disease (imbalanced)
- Missing values: Present in various columns (handled via imputation)

### Data Quality

- Missing values are filled using median (numerical) and mode (categorical)
- Class imbalance is addressed using SMOTE (Synthetic Minority Over-sampling Technique)
- Features are standardized using StandardScaler for appropriate models

---

## Legacy Dataset: heart.csv

### Overview
This dataset contains 303 patient records with 13 clinical features used to predict the presence of heart disease.

### Features

1. **age**: Age in years (29-77)
2. **sex**: Sex (0 = female, 1 = male)
3. **cp**: Chest pain type (0-3)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (mm Hg) (94-200)
5. **chol**: Serum cholesterol (mg/dl) (126-564)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
   - 0: Normal
   - 1: ST-T wave abnormality
   - 2: Left ventricular hypertrophy
8. **thalach**: Maximum heart rate achieved (71-202)
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest (0-6.2)
11. **slope**: Slope of the peak exercise ST segment (0-2)
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
12. **ca**: Number of major vessels colored by fluoroscopy (0-3)
13. **thal**: Thalassemia (0-3)
    - 0: Normal
    - 1: Fixed defect
    - 2: Reversible defect
    - 3: Not described

### Target Variable

**target**: Diagnosis of heart disease
- 0: No disease (< 50% diameter narrowing)
- 1: Disease present (> 50% diameter narrowing)

### Dataset Statistics

- Total samples: 303
- Features: 13
- Target classes: 2 (Binary classification)
- Class distribution: ~29% No disease, ~71% Disease

---

## Source

Both datasets are synthetic datasets generated for educational purposes.
