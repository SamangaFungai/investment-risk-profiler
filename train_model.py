"""
train_model.py
---------------
Trains an Investment Risk Profiling classifier.

Pipeline:
  1. Load data (generates it first if investor_data.csv doesn't exist)
  2. Preprocess (train/test split + feature scaling)
  3. Train a RandomForestClassifier
  4. Evaluate (accuracy, precision/recall/F1, confusion matrix)
  5. Save the trained model + scaler to model.pkl for the Streamlit app
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from generate_data import generate_dataset

DATA_PATH = "investor_data.csv"
MODEL_PATH = "model.pkl"

FEATURES = [
    "Age",
    "Annual_Income",
    "Investment_Amount",
    "Years_Experience",
    "Financial_Knowledge_Score",
    "Portfolio_Diversity_Score",
    "Debt_to_Income_Ratio",
]
TARGET = "Risk_Category"


def load_data():
    if not os.path.exists(DATA_PATH):
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def main():
    # 1. Load data ---------------------------------------------------------
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    # 2. Preprocessing -------------------------------------------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train model -----------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # 4. Evaluate ----------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    print("=" * 60)
    print(f"Test Accuracy:        {acc:.4f}")
    print(f"5-Fold CV Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print("=" * 60)
    print("\nClassification Report:\n")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoder.classes_
        )
    )
    print("Confusion Matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=[f"true_{c}" for c in label_encoder.classes_],
        columns=[f"pred_{c}" for c in label_encoder.classes_],
    ))

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances)

    # 5. Save model + scaler + encoder for deployment ----------------------
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "features": FEATURES,
            "test_accuracy": acc,
        },
        MODEL_PATH,
    )
    print(f"\nSaved trained model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
