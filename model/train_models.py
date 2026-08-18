"""
ML Assignment 2 - Dry Bean multi-class classification.
BITS ID: 2025AC05211

Trains five classifiers on the UCI Dry Bean dataset, scores them on a held-out
test split, and writes the fitted pipelines + metrics into this folder so that
the Streamlit app can load them without retraining.

Run:  python model/train_models.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_CSV = os.path.join(ROOT, "data", "dry_bean.csv")
TEST_CSV = os.path.join(ROOT, "test_data.csv")
TARGET = "Class"
SEED = 2025  # kept fixed so the numbers in README.md are reproducible


def build_estimators():
    """One scaled pipeline per algorithm. Scaling matters for LR / kNN, is
    harmless for the tree-based ones, and keeps the app's predict path uniform."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    DecisionTreeClassifier(
                        criterion="entropy",
                        max_depth=12,
                        min_samples_leaf=8,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance", p=2)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [("scaler", StandardScaler()), ("clf", GaussianNB(var_smoothing=1e-9))]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=1,
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def score(y_true, y_pred, y_proba, n_classes):
    """Macro-averaged metrics - the Dry Bean classes are imbalanced (BOMBAY has
    522 rows, DERMASON 3546), so macro treats every bean type equally."""
    average = "macro"
    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = pd.read_csv(RAW_CSV)
    print(f"dataset: {df.shape[0]} rows x {df.shape[1] - 1} features")
    print(df[TARGET].value_counts().to_string())

    X = df.drop(columns=[TARGET])
    y_raw = df[TARGET]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    n_classes = len(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"train {X_train.shape[0]} / test {X_test.shape[0]}")

    # The test split is what gets pushed to GitHub and uploaded in the app.
    test_frame = X_test.copy()
    test_frame[TARGET] = encoder.inverse_transform(y_test)
    test_frame.to_csv(TEST_CSV, index=False)
    print(f"wrote {TEST_CSV}")

    results = {}
    for name, pipe in build_estimators().items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_secs = time.perf_counter() - t0

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        metrics = score(y_test, y_pred, y_proba, n_classes)
        metrics["TrainSeconds"] = round(fit_secs, 3)
        results[name] = metrics

        slug = name.lower().replace(" ", "_")
        joblib.dump(pipe, os.path.join(HERE, f"{slug}.joblib"), compress=3)
        print(
            f"{name:<20} acc={metrics['Accuracy']:.4f} auc={metrics['AUC']:.4f} "
            f"f1={metrics['F1']:.4f} mcc={metrics['MCC']:.4f} ({fit_secs:.2f}s)"
        )

    joblib.dump(encoder, os.path.join(HERE, "label_encoder.joblib"))

    meta = {
        "dataset": "UCI Dry Bean Dataset (ID 602)",
        "source": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
        "n_rows": int(df.shape[0]),
        "n_features": int(df.shape[1] - 1),
        "feature_names": list(X.columns),
        "classes": list(encoder.classes_),
        "class_counts": {k: int(v) for k, v in y_raw.value_counts().items()},
        "test_size": 0.2,
        "random_state": SEED,
        "averaging": "macro (One-vs-Rest for AUC)",
        "metrics": results,
    }
    with open(os.path.join(HERE, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    table = pd.DataFrame(results).T[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    print("\n" + table.round(4).to_markdown())


if __name__ == "__main__":
    main()
