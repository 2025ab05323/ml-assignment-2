# ml-assignment-2
Machine Learning Assignment-2
# Machine Learning Classification Models - Assignment 2

**Student Name:** Shailendra Mohan Katiyar 
**BITS ID:** 2025AB05323
**Course:** AIMLCZG565  
**Email:** 2025AB05323@WILP.BITS-PILANI.ac.in
**GitHub:** https://github.com/2025ab05323/ml-Assignment-2
**Assignment:** Assignment 2 - Classification Models and Streamlit Deployment

---

## 📋 Problem Statement

This project implements and compares six different machine learning classification models on Heart Disease Dataset. The objective is to:

1. Build and train multiple classification algorithms
2. Evaluate their performance using comprehensive metrics
3. Deploy an interactive web application for model demonstration
4. Compare model performances to identify the best approach

This project is for predicting heart disease based on patient medical records.

---

## 📊 Dataset Description

**Dataset Name:** Heart Disease
**Source:** https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

### Dataset Overview:

- **Total Instances:** 1025
- **Total Features:** 13
- **Target Variable:** target
- **Classification Type:** Binary  
- **Class Distribution:** target
1    526
0    499
Name: count, dtype: int64

### Features:
#   Column    Dtype  
---  ------    -----  
 0   age       int64  
 1   sex       int64  
 2   cp        int64  
 3   trestbps  int64  
 4   chol      int64  
 5   fbs       int64  
 6   restecg   int64  
 7   thalach   int64  
 8   exang     int64  
 9   oldpeak   float64
 10  slope     int64  
 11  ca        int64  
 12  thal      int64  
 13  target    int64  
 

### Data Preprocessing:
- Missing values handled using [method - mean imputation]
- Categorical variables encoded using [method - one-hot encoding]
- Features scaled using StandardScaler
- Train-test split: 80-20 ratio with stratification

---

## Models Used

### Comparison Table
Model|Accuracy|AUC|Precision|Recall|F1|MCC
Logistic Regression|0.8097560975609757|0.9298095238095239|0.8224760728619943|0.8097560975609757|0.807243989148881|0.630908308763638
Decision Tree|0.8780487804878049|0.9597142857142857|0.8784596955050075|0.8780487804878049|0.8780661933841442|0.7564505687994391
K-Nearest Neighbors|1.0|1.0|1.0|1.0|1.0|1.0
Naive Bayes|0.8292682926829268|0.9042857142857142|0.8314689161929212|0.8292682926829268|0.8287540036699943|0.6601634114374199
Random Forest|0.9560975609756097|0.9898095238095238|0.956457966641522|0.9560975609756097|0.9560724505507727|0.9124719931456753
XGBoost|1.0|1.0|1.0|1.0|1.0|1.0

---

## Model Performance Observations

### 1. Logistic Regression
**Performance Analysis:**
Achieved 80.98% accuracy with an F1-score of 0.807, showing moderate performance.
MCC of 0.63 indicates a strong correlation between predictions and actual values.
Works well when relationships are linear and features are independent.
Strengths: Fast training, interpretable coefficients.
Limitations: May underperform on complex non-linear data.

### 2. Decision Tree
**Performance Analysis:**
Accuracy of 87.80% and F1-score of 0.878 reflect solid predictive ability.
MCC of 0.75 shows strong correlation.
Strengths: Easy to interpret, captures non-linear relationships.
Limitations: Can overfit without pruning or ensemble methods.

### 3. K-Nearest Neighbors (KNN)
**Performance Analysis:**
Perfect scores across all metrics (100% accuracy, F1, MCC = 1.0).
Suggests possible overfitting or dataset being too simple.
Strengths: Simple, non-parametric.
Limitations: Computationally expensive on large datasets, sensitive to noise.

### 4. Naive Bayes (Gaussian)
**Performance Analysis:**
Accuracy of 82.93% with F1-score of 0.829.
MCC of 0.66 indicates strong correlation.
Strengths: Fast, works well with high-dimensional data.
Limitations: Assumes feature independence, which may not hold.

### 5. Random Forest (Ensemble)
**Performance Analysis:**
Accuracy of 95.61%, F1-score of 0.956, MCC of 0.91 → very strong performance.
AUC of 0.99 shows excellent class separation.
Strengths: Handles non-linear patterns, reduces overfitting via ensembling.
Limitations: Less interpretable compared to single trees.

### 6. XGBoost (Ensemble)
**Performance Analysis:**
Perfect scores (100% accuracy, F1, MCC = 1.0).
Indicates possible overfitting or dataset being too easy.
Strengths: Highly powerful, optimized boosting algorithm.
Limitations: Computationally intensive, may overfit if not tuned.

---
