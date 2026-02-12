import os
import sys

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data_loader import load_training_data
from src.features import build_preprocessing_pipeline

MODEL_MAP = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
}


def train_model(config):
    """Train a classification model and save it to disk."""
    X, y = load_training_data(config)

    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model_type = config["model"]["type"]
    model_params = config["model"].get("params", {})

    if model_type not in MODEL_MAP:
        sys.exit(
            f"Error: Unknown model type '{model_type}'. "
            f"Choose from: {list(MODEL_MAP.keys())}"
        )

    classifier = MODEL_MAP[model_type](**model_params)
    preprocessing = build_preprocessing_pipeline(config)

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])

    print(f"Training {model_type} on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)

    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")

    output_dir = config["model"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
