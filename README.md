# Credit Card Fraud Detection System

A machine learning system that detects fraudulent credit card transactions in real-time, addressing the core challenge of extreme class imbalance (only 0.17% of transactions are fraudulent).

## Problem
With fraud representing a tiny fraction of transactions, naive models can achieve 99.8% accuracy while catching zero fraud. This project focuses on proper handling of imbalanced data and evaluation using precision/recall rather than misleading accuracy metrics.

## Approach
- **Dataset**: 284,807 anonymized transactions (PCA-transformed features V1-V28 + Amount)
- **Imbalance handling**: SMOTE (Synthetic Minority Oversampling) applied only to training data
- **Models compared**: Logistic Regression, Random Forest, XGBoost
- **Evaluation**: Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

## Results
| Model | Precision (Fraud) | Recall (Fraud) | F1-Score |
|---|---|---|---|
| Logistic Regression | 0.06 | 0.92 | 0.11 |
| **Random Forest** | **0.87** | **0.83** | **0.85** |
| XGBoost | 0.69 | 0.86 | 0.76 |

**Random Forest** was selected as the final model — best precision-recall balance, minimizing false alarms while catching the majority of fraud cases.

## Deployment
Deployed as a Django web application with a real-time transaction scoring dashboard — input transaction features and get an instant fraud probability + flag.

## Tech Stack
Python, scikit-learn, XGBoost, imbalanced-learn, Django, Pandas

## How to Run
```bash
pip install -r requirements.txt
python manage.py runserver
```

## Key Learning
Fraud detection isn't about accuracy — it's about the precision-recall tradeoff. A model with 92% recall but 6% precision (Logistic Regression) is operationally useless due to false alarm volume, while Random Forest's balanced approach makes it production-viable.
