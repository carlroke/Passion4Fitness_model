import os
import sys

import joblib
import pandas as pd


def predict(config, input_path):
    """Load a saved model and run predictions on a new CSV file."""
    model_path = os.path.join(config["model"]["output_dir"], "model.joblib")
    if not os.path.isfile(model_path):
        sys.exit(
            f"Error: No saved model found at '{model_path}'. Run 'train' first."
        )

    if not os.path.isfile(input_path):
        sys.exit(f"Error: Input file '{input_path}' not found.")

    pipeline = joblib.load(model_path)
    feature_columns = config["data"]["feature_columns"]

    df = pd.read_csv(input_path)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        sys.exit(
            f"Error: Missing feature columns in input: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    X = df[feature_columns]
    predictions = pipeline.predict(X)

    df["prediction"] = predictions
    print(df[["prediction"]].to_string(index=False))
    print(f"\n{len(predictions)} predictions generated.")
