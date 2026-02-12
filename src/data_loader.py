import os
import sys

import pandas as pd


def load_training_data(config):
    """Load all CSVs from the training directory, validate columns, return X and y."""
    training_dir = config["data"]["training_dir"]
    feature_columns = config["data"]["feature_columns"]
    target_column = config["data"]["target_column"]

    if not os.path.isdir(training_dir):
        sys.exit(f"Error: Training directory '{training_dir}' does not exist.")

    csv_files = sorted(
        os.path.join(training_dir, f)
        for f in os.listdir(training_dir)
        if f.endswith(".csv")
    )

    if not csv_files:
        sys.exit(
            f"Error: No CSV files found in '{training_dir}'. "
            "Place your labeled training data there and try again."
        )

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(csv_files)} file(s), {len(df)} total rows.")

    # Validate feature columns
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        sys.exit(
            f"Error: Missing feature columns in data: {missing_features}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Validate target column
    if target_column not in df.columns:
        sys.exit(
            f"Error: Target column '{target_column}' not found in data.\n"
            f"Available columns: {list(df.columns)}\n\n"
            "Your CSV files need a label column for supervised learning.\n"
            f"Add a '{target_column}' column to your data with values like:\n"
            "  - Form quality:    good / fair / poor\n"
            "  - Rider position:  seated / standing / sprinting\n"
            "  - L/R balance:     balanced / left_dominant / right_dominant"
        )

    X = df[feature_columns]
    y = df[target_column]
    return X, y
