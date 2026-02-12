import os
import sys

import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data_loader import load_training_data


def evaluate_model(config):
    """Load a saved model and evaluate it on the test split."""
    model_path = os.path.join(config["model"]["output_dir"], "model.joblib")
    if not os.path.isfile(model_path):
        sys.exit(
            f"Error: No saved model found at '{model_path}'. Run 'train' first."
        )

    pipeline = joblib.load(model_path)

    X, y = load_training_data(config)
    test_size = config["training"]["test_size"]
    random_state = config["training"]["random_state"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    y_pred = pipeline.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
