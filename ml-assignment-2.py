#========================================================
#MACHINE LEARNING ASSIGNMENT 2 - COMPLETE PROGRAM
#========================================================

#================================================================================================
# STEP 1.3: Load and Explore Dataset
#================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import xgboost

# Load dataset
df = pd.read_csv('heart.csv')

# Basic exploration
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head(10))

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nTarget Variable Distribution:")
print(df['target'].value_counts())

# Visualize target distribution
plt.figure(figsize=(8, 6))
df['target'].value_counts().plot(kind='bar')
plt.title('Target Variable Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('target_distribution.png')
plt.close()

"""
=============================================================================
PHASE 2: DATA PREPROCESSING
=============================================================================
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# STEP 2.1: Handle Missing Values
# --------------------------------
# For numerical columns: replaces all missing numeric values with mean/median values
df.fillna(df.mean(), inplace=True)

# For categorical columns: replaces all missing non numeric values with object values
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# STEP 2.2: Encode Categorical Variables
# ---------------------------------------
# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Remove target if it's in the list
if 'target' in categorical_cols:
    categorical_cols.remove('target')

# Use one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Encode target if it's categorical
le = LabelEncoder()
if df['target'].dtype == 'object':
    df['target'] = le.fit_transform(df['target'])

# STEP 2.3: Split Features and Target
# ------------------------------------
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")
print(f"Number of Classes: {len(y.unique())}")

# STEP 2.4: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Ensures balanced split
)

print(f"\nTraining Set: {X_train.shape[0]} samples")
print(f"Test Set: {X_test.shape[0]} samples")

# STEP 2.5: Feature Scaling
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use in Streamlit
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n Data preprocessing completed!")

"""
=============================================================================
PHASE 3: MODEL IMPLEMENTATION
=============================================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Initialize results storage
results = []

# STEP 3.1: Logistic Regression
# ------------------------------
print("\n" + "="*70)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*70)

lr_model = LogisticRegression(
    max_iter=1000, 
    random_state=42,
    solver='lbfgs'  # Good for multiclass
)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)

# Calculate metrics
lr_results = {
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, lr_pred),
    'Precision': precision_score(y_test, lr_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, lr_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test, lr_pred, average='weighted', zero_division=0),
    'MCC': matthews_corrcoef(y_test, lr_pred)
}

# AUC Score
try:
    if len(np.unique(y_test)) == 2:
        lr_results['AUC'] = roc_auc_score(y_test, lr_pred_proba[:, 1])
    else:
        lr_results['AUC'] = roc_auc_score(y_test, lr_pred_proba, 
                                         multi_class='ovr', average='weighted')
except:
    lr_results['AUC'] = 0.0

results.append(lr_results)

# Save model
with open('models/logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

print("\nMetrics:")
for metric, value in lr_results.items():
    if metric != 'Model':
        print(f"  {metric}: {value:.4f}")

# STEP 3.2: Decision Tree
# ------------------------
print("\n" + "="*70)
print("MODEL 2: DECISION TREE CLASSIFIER")
print("="*70)

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,  # Prevent overfitting
    min_samples_split=20,
    min_samples_leaf=10
)
dt_model.fit(X_train_scaled, y_train)

dt_pred = dt_model.predict(X_test_scaled)
dt_pred_proba = dt_model.predict_proba(X_test_scaled)

dt_results = {
    'Model': 'Decision Tree',
    'Accuracy': accuracy_score(y_test, dt_pred),
    'Precision': precision_score(y_test, dt_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, dt_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test, dt_pred, average='weighted', zero_division=0),
    'MCC': matthews_corrcoef(y_test, dt_pred)
}

try:
    if len(np.unique(y_test)) == 2:
        dt_results['AUC'] = roc_auc_score(y_test, dt_pred_proba[:, 1])
    else:
        dt_results['AUC'] = roc_auc_score(y_test, dt_pred_proba, 
                                         multi_class='ovr', average='weighted')
except:
    dt_results['AUC'] = 0.0

results.append(dt_results)

with open('models/decision_tree.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

print("\nMetrics:")
for metric, value in dt_results.items():
    if metric != 'Model':
        print(f"  {metric}: {value:.4f}")

# STEP 3.3: K-Nearest Neighbors
# ------------------------------
print("\n" + "="*70)
print("MODEL 3: K-NEAREST NEIGHBORS")
print("="*70)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # Weight by distance
    metric='euclidean'
)
knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)
knn_pred_proba = knn_model.predict_proba(X_test_scaled)

knn_results = {
    'Model': 'K-Nearest Neighbors',
    'Accuracy': accuracy_score(y_test, knn_pred),
    'Precision': precision_score(y_test, knn_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, knn_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test, knn_pred, average='weighted', zero_division=0),
    'MCC': matthews_corrcoef(y_test, knn_pred)
}

try:
    if len(np.unique(y_test)) == 2:
        knn_results['AUC'] = roc_auc_score(y_test, knn_pred_proba[:, 1])
    else:
        knn_results['AUC'] = roc_auc_score(y_test, knn_pred_proba, 
                                          multi_class='ovr', average='weighted')
except:
    knn_results['AUC'] = 0.0

results.append(knn_results)

with open('models/k_nearest_neighbors.pkl', 'wb') as f:
    pickle.dump(knn_model, f)

print("\nMetrics:")
for metric, value in knn_results.items():
    if metric != 'Model':
        print(f"  {metric}: {value:.4f}")

# STEP 3.4: Naive Bayes
# ----------------------
print("\n" + "="*70)
print("MODEL 4: NAIVE BAYES (GAUSSIAN)")
print("="*70)

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

nb_pred = nb_model.predict(X_test_scaled)
nb_pred_proba = nb_model.predict_proba(X_test_scaled)

nb_results = {
    'Model': 'Naive Bayes',
    'Accuracy': accuracy_score(y_test, nb_pred),
    'Precision': precision_score(y_test, nb_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, nb_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test, nb_pred, average='weighted', zero_division=0),
    'MCC': matthews_corrcoef(y_test, nb_pred)
}

try:
    if len(np.unique(y_test)) == 2:
        nb_results['AUC'] = roc_auc_score(y_test, nb_pred_proba[:, 1])
    else:
        nb_results['AUC'] = roc_auc_score(y_test, nb_pred_proba, 
                                         multi_class='ovr', average='weighted')
except:
    nb_results['AUC'] = 0.0

results.append(nb_results)

with open('models/naive_bayes.pkl', 'wb') as f:
    pickle.dump(nb_model, f)

print("\nMetrics:")
for metric, value in nb_results.items():
    if metric != 'Model':
        print(f"  {metric}: {value:.4f}")

# STEP 3.5: Random Forest
# ------------------------
print("\n" + "="*70)
print("MODEL 5: RANDOM FOREST (ENSEMBLE)")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    n_jobs=-1  # Use all CPU cores
)
rf_model.fit(X_train_scaled, y_train)

rf_pred = rf_model.predict(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test_scaled)

rf_results = {
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, rf_pred),
    'Precision': precision_score(y_test, rf_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, rf_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test, rf_pred, average='weighted', zero_division=0),
    'MCC': matthews_corrcoef(y_test, rf_pred)
}

try:
    if len(np.unique(y_test)) == 2:
        rf_results['AUC'] = roc_auc_score(y_test, rf_pred_proba[:, 1])
    else:
        rf_results['AUC'] = roc_auc_score(y_test, rf_pred_proba, 
                                         multi_class='ovr', average='weighted')
except:
    rf_results['AUC'] = 0.0

results.append(rf_results)

with open('models/random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("\nMetrics:")
for metric, value in rf_results.items():
    if metric != 'Model':
        print(f"  {metric}: {value:.4f}")

# STEP 3.6: XGBoost
# -----------------
print("\n" + "="*70)
print("MODEL 6: XGBOOST (ENSEMBLE)")
print("="*70)

xgb_model = XGBClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)

xgb_results = {
    'Model': 'XGBoost',
    'Accuracy': accuracy_score(y_test, xgb_pred),
    'Precision': precision_score(y_test, xgb_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, xgb_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test, xgb_pred, average='weighted', zero_division=0),
    'MCC': matthews_corrcoef(y_test, xgb_pred)
}

try:
    if len(np.unique(y_test)) == 2:
        xgb_results['AUC'] = roc_auc_score(y_test, xgb_pred_proba[:, 1])
    else:
        xgb_results['AUC'] = roc_auc_score(y_test, xgb_pred_proba, 
                                          multi_class='ovr', average='weighted')
except:
    xgb_results['AUC'] = 0.0

results.append(xgb_results)

with open('models/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("\nMetrics:")
for metric, value in xgb_results.items():
    if metric != 'Model':
        print(f"  {metric}: {value:.4f}")

"""
=============================================================================
PHASE 4: RESULTS COMPILATION
=============================================================================
"""

# Create results DataFrame
results_df = pd.DataFrame(results)

# Reorder columns
column_order = ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
results_df = results_df[column_order]

# Display comparison table
print("\n" + "="*80)
print("FINAL COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Save to CSV
results_df.to_csv('model_comparison.csv', index=False)
print("\n Results saved to 'model_comparison.csv'")

# Save test data for Streamlit
test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
test_data['target'] = y_test.values
test_data.to_csv('test_data.csv', index=False)
print(" Test data saved to 'test_data.csv'")

#-==========================================================================

