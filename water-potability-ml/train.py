import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

C = 1.0

# Data preparation
df = pd.read_csv("data/water_potability.csv")
df.columns = df.columns.str.lower()
df

#  Checking Class Imbalance
print(df["potability"].value_counts())
print(df["potability"].value_counts(normalize=True))


# Imputation
imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(
    imputer.fit_transform(df.drop("potability", axis=1)),
    columns=df.drop("potability", axis=1).columns,
)
df_imputed


# Data splitting
X = df_imputed
y = df["potability"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Training the model
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=5000),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc_roc": roc_auc_score(y_val, y_proba),
    }


# Apply SMOTE to balance classes
smote = SMOTE(
    random_state=42,
    k_neighbors=5,
)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
    ),
}


results = {}
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "auc_roc": roc_auc_score(y_val, y_proba),
    }


# Create XGBoost model with optimized parameters
xgb_model = XGBClassifier(
    n_estimators=300,  # Number of boosting rounds
    max_depth=6,  # Maximum tree depth
    learning_rate=0.1,  # Step size shrinkage (eta)
    subsample=0.8,  # Fraction of samples used per tree
    colsample_bytree=0.8,  # Fraction of features used per tree
    scale_pos_weight=1.56,  # Handle class imbalance
    gamma=0,  # Minimum loss reduction for split
    min_child_weight=1,  # Minimum sum of instance weight in child
    reg_alpha=0,  # L1 regularization
    reg_lambda=1,  # L2 regularization
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",  # Evaluation metric
)

# Train the model on validation dataset
xgb_model.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# Make predictions on the validation dataset
y_pred_xgb = xgb_model.predict(X_val)
y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]


# Create XGBoost model with optimized parameters
xgb_model = XGBClassifier(
    n_estimators=300,  # Number of boosting rounds
    max_depth=6,  # Maximum tree depth
    learning_rate=0.1,  # Step size shrinkage (eta)
    subsample=0.8,  # Fraction of samples used per tree
    colsample_bytree=0.8,  # Fraction of features used per tree
    scale_pos_weight=1.56,  # Handle class imbalance
    gamma=0,  # Minimum loss reduction for split
    min_child_weight=1,  # Minimum sum of instance weight in child
    reg_alpha=0,  # L1 regularization
    reg_lambda=1,  # L2 regularization
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",  # Evaluation metric
)

# Train the final model on the test dataset
xgb_model.fit(
    X_train_balanced,
    y_train_balanced,
    eval_set=[(X_test, y_test)],
    verbose=False,  # Set to True to see training progress
)

# Make predictions on the test dataset
y_pred_xgb = xgb_model.predict(X_test)
# y_proba_xgb = xgb_model.predict_proba(X_test)[0, 1]


# Saving the model to pickle
output_file = "model_C=%s.bin" % C
output_file

with open(output_file, "wb") as f_out:
    pickle.dump(xgb_model, f_out)
