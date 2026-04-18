"""Task 3: Train a fraud detection model and report accuracy and precision."""

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = ["Claim_Amount", "Claim_Type", "Location", "Previous_Claims"]
TARGET_COLUMN = "Fraud"


def build_pipeline() -> Pipeline:
    numeric_features = ["Claim_Amount", "Previous_Claims"]
    categorical_features = ["Claim_Type", "Location"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def normalize_target(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)


def extract_feature_importance(model: Pipeline) -> Dict[str, float]:
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    weights = classifier.coef_[0]
    pairs = sorted(zip(feature_names, weights), key=lambda item: abs(item[1]), reverse=True)
    return {name: float(weight) for name, weight in pairs[:10]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 3: Train a fraud detection model from CSV data.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--model-output", type=Path, default=Path("model.pkl"))
    parser.add_argument("--metrics-output", type=Path, default=Path("model_metrics.json"))
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {missing}")

    X = data[FEATURE_COLUMNS].copy()
    y = normalize_target(data[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "top_feature_weights": extract_feature_importance(model),
    }

    joblib.dump(model, args.model_output)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
