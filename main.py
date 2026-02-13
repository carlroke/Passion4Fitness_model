import argparse
import yaml

from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict
from src.discover import discover


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Passion4Fitness — Cycle Power Meter ML Pipeline"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Train a model on labeled CSV data")
    subparsers.add_parser("evaluate", help="Evaluate the saved model on test data")

    discover_parser = subparsers.add_parser(
        "discover", help="Data discovery on raw CSV output files"
    )
    discover_parser.add_argument(
        "--data-dir", default=None, help="Path to CSV directory (default: Passion4Fitness DATA_OUTPUT)"
    )

    predict_parser = subparsers.add_parser(
        "predict", help="Run predictions on new CSV data"
    )
    predict_parser.add_argument(
        "--input", required=True, help="Path to input CSV file"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "discover":
        data_dir = args.data_dir if args.data_dir else None
        discover(data_dir) if data_dir else discover()
    elif args.command == "train":
        train_model(config)
    elif args.command == "evaluate":
        evaluate_model(config)
    elif args.command == "predict":
        predict(config, args.input)


if __name__ == "__main__":
    main()
