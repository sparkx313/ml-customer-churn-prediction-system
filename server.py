"""Train and serve a customer-churn model from a supplied CSV."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ChurnError(ValueError):
    """Raised when the training data or requested model is invalid."""


class PredictionRequest(BaseModel):
    features: dict[str, Any]


def train_model(
    data_path: Path,
    target_name: str,
    model_path: Path,
    positive_label: str | None = None,
    threshold: float = 0.5,
    drop_columns: list[str] | None = None,
) -> dict:
    source = data_path.expanduser().resolve()
    destination = model_path.expanduser().resolve()
    if not source.is_file():
        raise ChurnError(f"Training CSV does not exist: {source}")
    if source == destination:
        raise ChurnError("Model output must be different from the training CSV.")
    if not 0 < threshold < 1:
        raise ChurnError("Decision threshold must be strictly between 0 and 1.")

    try:
        frame = pd.read_csv(source, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ChurnError(f"Could not read training CSV: {exc}") from exc
    if frame.empty:
        raise ChurnError("Training CSV has no data rows.")

    normalized = [str(column).strip().casefold() for column in frame.columns]
    if len(set(normalized)) != len(normalized):
        raise ChurnError("Training CSV contains duplicate column names after normalization.")
    target_matches = [column for column in frame.columns if str(column).strip().casefold() == target_name.strip().casefold()]
    if len(target_matches) != 1:
        raise ChurnError(
            f"Target column {target_name!r} was not found exactly once. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )
    target_column = target_matches[0]

    requested_drops = [item.strip().casefold() for item in drop_columns or [] if item.strip()]
    drop_matches = {
        column for column in frame.columns
        if str(column).strip().casefold() in requested_drops
    }
    if target_column in drop_matches:
        raise ChurnError("Target column cannot also be dropped.")
    feature_columns = [column for column in frame.columns if column != target_column and column not in drop_matches]
    if not feature_columns:
        raise ChurnError("No feature columns remain after excluding the target and requested drops.")

    labels = frame[target_column].astype("string").str.strip()
    valid_target = labels.notna() & labels.ne("").fillna(False)
    frame = frame.loc[valid_target, feature_columns].copy()
    labels = labels.loc[valid_target]
    classes = sorted(str(value) for value in labels.unique())
    if len(classes) != 2:
        raise ChurnError(f"Target must contain exactly two non-empty classes; found {len(classes)}.")

    if positive_label is not None:
        matches = [value for value in classes if value.casefold() == positive_label.strip().casefold()]
        if len(matches) != 1:
            raise ChurnError(f"Positive label {positive_label!r} was not found in target classes {classes}.")
        positive = matches[0]
    else:
        preferred = {"1", "yes", "true", "churn", "churned"}
        positive = next((value for value in classes if value.casefold() in preferred), classes[-1])
    negative = next(value for value in classes if value != positive)
    y = (labels.astype(str) == positive).astype("int64")

    if y.value_counts().min() < 2:
        raise ChurnError("Each target class needs at least two rows for a stratified holdout split.")
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            frame,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError as exc:
        raise ChurnError(f"Could not create a stratified holdout split: {exc}") from exc

    numeric_columns = list(x_train.select_dtypes(include=["number", "bool"]).columns)
    categorical_columns = [column for column in x_train.columns if column not in numeric_columns]
    for column in numeric_columns:
        x_train[column] = x_train[column].replace([np.inf, -np.inf], np.nan)
        x_test[column] = x_test[column].replace([np.inf, -np.inf], np.nan)
    empty_features = [column for column in x_train.columns if x_train[column].isna().all()]
    if empty_features:
        raise ChurnError(
            "These features are completely missing in the training split; exclude them with --drop-columns: "
            + ", ".join(map(str, empty_features))
        )
    for column in categorical_columns:
        x_train[column] = x_train[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
        x_test[column] = x_test[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    try:
        pipeline.fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
    except (ValueError, TypeError) as exc:
        raise ChurnError(f"Could not fit the churn pipeline: {exc}") from exc

    predictions = (probabilities >= threshold).astype("int64")
    metrics = {
        "holdout_rows": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_for_positive_class": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_for_positive_class": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_for_positive_class": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "positive_rate_in_holdout": float(y_test.mean()),
    }
    artifact = {
        "pipeline": pipeline,
        "feature_columns": [str(column) for column in feature_columns],
        "numeric_features": [str(column) for column in numeric_columns],
        "categorical_features": [str(column) for column in categorical_columns],
        "target_column": str(target_column),
        "positive_label": positive,
        "negative_label": negative,
        "threshold": threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(x_train)),
        "metrics": metrics,
    }
    _atomic_model_save(artifact, destination)
    return {
        "model_file": str(destination),
        "target_column": str(target_column),
        "positive_label": positive,
        "negative_label": negative,
        "training_rows": int(len(x_train)),
        "feature_columns": [str(column) for column in feature_columns],
        "dropped_columns": [str(column) for column in drop_matches],
        "threshold": threshold,
        "metrics": metrics,
        "limitations": [
            "Holdout metrics estimate performance only for this split and dataset.",
            "Class imbalance and threshold choice affect precision and recall.",
            "Predictions are not causal explanations and should not be used as the sole basis for customer decisions.",
        ],
    }


def _atomic_model_save(artifact: dict, destination: Path) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".joblib",
            dir=destination.parent, delete=False,
        ) as temp_file:
            temporary = Path(temp_file.name)
        joblib.dump(artifact, temporary)
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise ChurnError(f"Could not save model artifact: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


@lru_cache(maxsize=4)
def _load_artifact(path: str) -> dict:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    try:
        artifact = joblib.load(model_path)
    except Exception as exc:
        raise ValueError(f"Could not load trusted model artifact: {exc}") from exc
    required = {"pipeline", "feature_columns", "positive_label", "negative_label", "threshold"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Model artifact has an unsupported format.")
    return artifact


def _current_model() -> tuple[str, dict]:
    path = str(Path(os.environ.get("CHURN_MODEL_PATH", "churn_model.joblib")).expanduser().resolve())
    try:
        return path, _load_artifact(path)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model is not ready: {exc}") from exc


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    path, artifact = _current_model()
    return {
        "status": "ready",
        "model_file": Path(path).name,
        "trained_at": artifact.get("trained_at"),
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    _, artifact = _current_model()
    columns = artifact["feature_columns"]
    provided = set(request.features)
    expected = set(columns)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )
    values = dict(request.features)
    for column in artifact.get("numeric_features", []):
        value = values[column]
        if value is None:
            values[column] = np.nan
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Feature {column!r} must be numeric or null.") from exc
        if not np.isfinite(numeric_value):
            raise HTTPException(status_code=422, detail=f"Feature {column!r} must be finite.")
        values[column] = numeric_value
    for column in artifact.get("categorical_features", []):
        value = values[column]
        if value is None:
            values[column] = np.nan
            continue
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise HTTPException(status_code=422, detail=f"Feature {column!r} must be a scalar or null.")
        if value is not None and isinstance(value, float) and not np.isfinite(value):
            raise HTTPException(status_code=422, detail=f"Feature {column!r} must be finite.")
        if value is not None:
            values[column] = str(value)
    row = pd.DataFrame([{column: values[column] for column in columns}], columns=columns)
    try:
        probability = float(artifact["pipeline"].predict_proba(row)[0, 1])
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid feature value: {exc}") from exc
    threshold = float(artifact["threshold"])
    is_positive = probability >= threshold
    return {
        "probability": probability,
        "threshold": threshold,
        "prediction": artifact["positive_label"] if is_positive else artifact["negative_label"],
        "positive_label": artifact["positive_label"],
        "model_trained_at": artifact.get("trained_at"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or serve a customer-churn prediction model.")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train from a labeled CSV and save a local model")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--target", required=True, help="binary target column")
    train.add_argument("--model", type=Path, default=Path("churn_model.joblib"))
    train.add_argument("--positive-label", help="label treated as churn; inferred from common values by default")
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--drop-columns", default="", help="comma-separated IDs or columns to exclude from training")

    serve = commands.add_parser("serve", help="start the local prediction API")
    serve.add_argument("--model", type=Path, default=Path("churn_model.joblib"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train_model(
                args.data,
                target_name=args.target,
                model_path=args.model,
                positive_label=args.positive_label,
                threshold=args.threshold,
                drop_columns=args.drop_columns.split(","),
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        if not 1 <= args.port <= 65535:
            parser.error("port must be between 1 and 65535")
        model_path = args.model.expanduser().resolve()
        os.environ["CHURN_MODEL_PATH"] = str(model_path)
        uvicorn.run(app, host=args.host, port=args.port)
        return 0
    except ChurnError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
