# Passion4Fitness Model

Supervised machine learning pipeline for classifying cycling power meter data collected by the [Passion4Fitness](../Passion4Fitness) Android app.

## Project Structure

```
├── main.py              # CLI entry point (train / evaluate / predict / discover)
├── config.yaml          # Feature columns, model type, and training parameters
├── requirements.txt     # Python dependencies
├── src/
│   ├── data_loader.py   # Load and validate labeled training CSVs
│   ├── features.py      # Preprocessing pipeline (imputation + scaling)
│   ├── train.py         # Train a classifier and save to disk
│   ├── evaluate.py      # Evaluate saved model (classification report + confusion matrix)
│   ├── predict.py       # Run inference on new CSV data
│   └── discover.py      # Data discovery on raw CSV output files
├── data/training/       # Labeled training CSV files
└── models/              # Saved model artifacts (.joblib)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Data Discovery

Explore raw CSV data from the Passion4Fitness app before training:

```bash
python main.py discover
python main.py discover --data-dir /path/to/csv/directory
```

### Train

Train a model on labeled data in `data/training/`:

```bash
python main.py train
```

### Evaluate

Evaluate the trained model on a held-out test split:

```bash
python main.py evaluate
```

### Predict

Run predictions on a new CSV file:

```bash
python main.py predict --input path/to/data.csv
```

## Model

The pipeline uses a scikit-learn `Pipeline` with two stages:

1. **Preprocessing** — imputes missing values (mean strategy) and applies standard scaling (`SimpleImputer` + `StandardScaler`)
2. **Classifier** — one of three supported model types, configured via `model.type` in `config.yaml`:

| Model Type          | Class                        | Default |
|---------------------|------------------------------|---------|
| `random_forest`     | `RandomForestClassifier`     | Yes     |
| `gradient_boosting` | `GradientBoostingClassifier` |         |
| `svm`               | `SVC`                        |         |

The default configuration uses Random Forest with 100 estimators. Model parameters can be customised under `model.params` in `config.yaml`.

## Configuration

All settings are in `config.yaml`:

- **Feature columns**: `timestamp`, `instantaneous_power`, `left_power`, `right_power`, `pedal_power_balance`, `accumulated_torque`, `cumulative_crank_revs`, `last_crank_event_time`
- **Target column**: `label`
- **Model types**: `random_forest`, `gradient_boosting`, `svm`
- **Training**: configurable test split size and random state

## Data Format

Training CSVs must contain the feature columns listed in `config.yaml` plus a `label` column for classification targets (e.g. `balanced`, `left_dominant`, `right_dominant`).
