# 💰 Loan Approval Prediction — Machine Learning Coursework

> End-to-end machine learning pipeline predicting loan approval status and maximum lending amount, built for module 5DATA002W.2 – Machine Learning & Data Mining.

## 📖 Project Description

This project analyses a loan approval dataset to solve two related predictive problems:

1. **Classification** — Predict whether a loan application will be **Approved** or **Rejected**
2. **Regression** — For approved clients, predict the **Maximum Loan Amount** they should be offered

The primary business goal (per the coursework brief) is to maximise correct detection of **Rejected** applications to reduce the risk of future loan defaults — so **Recall on the Rejected class** is used as the key success metric throughout, alongside AUC-ROC.

## 📓 Notebooks

| Notebook | Purpose |
|---|---|
| `Notebook1_DataUnderstanding.ipynb` | Data exploration, cleaning, outlier removal, missing value imputation, encoding, and splitting into classification/regression datasets |
| `Notebook2_Classification.ipynb` | Builds and evaluates Naïve Bayes, Logistic Regression, and KNN classifiers; tunes KNN with GridSearchCV |
| `Notebook3_Regression_Ensemble.ipynb` | Builds a soft-voting ensemble (NB + LR) for classification, and Decision Tree Regressors (fully grown vs. pruned) to predict maximum loan amount |

## 🔄 Pipeline Overview
Raw Dataset (loan_approval_data.csv)
│
▼
Notebook 1: Cleaning & Preprocessing
├── Handle missing values (median/mean/mode imputation)
├── Remove impossible outliers (e.g. age = 123, negative loan amounts)
├── Label-encode categorical variables
├── Split into:
│     ├── loan_classification_data.csv  (all clients)
│     └── loan_regression_data.csv      (approved clients only)
│
▼
Notebook 2: Classification Models
├── Naïve Bayes
├── Logistic Regression
├── KNN (tuned via GridSearchCV, 5-fold CV, scored on Recall)
│
▼
Notebook 3: Ensemble & Regression
├── Part A: Soft-Voting Ensemble (NB + LR) for Approval Status
└── Part B: Decision Tree Regression for Max Loan Amount
├── DT-1: Fully grown tree
└── DT-2: Pruned tree (max_depth=4)

## 📊 Dataset

The dataset contains anonymised loan applicant records (age, income, employment length, loan intent, credit history, etc.). Two target variables are used:

- **Loan Approval Status** (classification target) — 0 = Approved, 1 = Rejected
- **Maximum Loan Amount** (regression target) — only available for approved applicants

*Note: Sex, Education, and Credit Application Acceptance fields are excluded from the dataset per GDPR anonymisation requirements stated in the coursework specification.*

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![scikit--learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat&logo=pandas&logoColor=white)

- **pandas / numpy** — data manipulation
- **scikit-learn** — modelling (Naïve Bayes, Logistic Regression, KNN, Decision Trees, Voting Ensemble, GridSearchCV)
- **plotly express / matplotlib** — visualisation (distributions, box plots, confusion matrices, ROC curves, decision trees)
- **Google Colab** — development environment (Google Drive used for data storage)

## 🚀 How to Run

These notebooks were developed in **Google Colab** and expect the dataset to be available in Google Drive.

1. Open a notebook in [Google Colab](https://colab.research.google.com/)
2. Update the file paths (`/content/drive/MyDrive/...`) to match your own Drive structure
3. Run `Notebook1` first to generate `loan_classification_data.csv` and `loan_regression_data.csv`
4. Run `Notebook2` for classification models
5. Run `Notebook3` for the ensemble model and regression trees

## 📈 Key Results Summary

- Models are compared primarily on **Recall (Rejected = 1)** and **AUC-ROC**, per the coursework's stated business priority of minimising missed high-risk (Reject) predictions
- KNN was hyperparameter-tuned via `GridSearchCV` (5-fold CV) across `n_neighbors` (1–24) and distance metric (Euclidean vs. Manhattan)
- A soft-voting ensemble combining Naïve Bayes and Logistic Regression was built and benchmarked against both individual base learners
- Two Decision Tree Regressors were compared for the maximum loan amount prediction: a fully grown tree vs. a pre-pruned tree (`max_depth=4`), evaluated on MAE, MSE, RMSE, and R²

## 👤 Author

**Mihiri Pabasara**
Module: 5DATA002W.2 – Machine Learning & Data Mining
