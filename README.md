# Travel Insurance Claim Prediction

## Project Introduction
This project builds a machine-learning risk-screening model to predict whether a travel insurance policy will result in a claim. The objective is to support underwriting and risk management by identifying high-risk policies early.

## Dataset
~44,000 policy records with policy, customer, trip, and product details. The target (`Claim`) is highly imbalanced (~1.5% positives).

## Approach
- Data cleaning and validation (handle negative and extreme values, treat rare categories)
- Feature engineering (duration categories, grouping rare destinations)
- Preprocessing and pipelines (ColumnTransformer + scikit-learn Pipelines)
- Model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Evaluation focused on Recall & Precision–Recall metrics
- Model interpretation via logistic regression coefficients and odds ratios

## Key Results
- Final model: **Logistic Regression (pipeline)**
- Stratified CV Recall: **~0.716 ± 0.038**
- PR-AUC: **~0.085** (improved slightly with engineered features)
- Primary drivers: destination, product type, sales agency

## Usage
- Notebook: `Faris Ridho_Capstone Project 3 Documentation.ipynb`
- Saved model: `logistic_regression_claim_model.pkl`
- To load the model:
```python
import pickle
with open('logistic_regression_claim_model.pkl','rb') as f:
    model = pickle.load(f)
preds = model.predict(X_new)