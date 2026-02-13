import os
import sys

import pandas as pd


DATA_OUTPUT_DIR = "/Users/rokec/PycharmProjects/Passion4Fitness/DATA_OUTPUT"


def discover(data_dir=DATA_OUTPUT_DIR):
    """Read all CSVs from the data output directory and print a data discovery summary."""
    if not os.path.isdir(data_dir):
        sys.exit(f"Error: Directory '{data_dir}' does not exist.")

    csv_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".csv")
    )
    if not csv_files:
        sys.exit(f"Error: No CSV files found in '{data_dir}'.")

    # Load non-empty files
    frames = []
    empty_files = []
    skipped_files = []
    for fname in csv_files:
        path = os.path.join(data_dir, fname)
        if os.path.getsize(path) == 0:
            empty_files.append(fname)
            continue
        try:
            frames.append((fname, pd.read_csv(path)))
        except Exception as e:
            skipped_files.append((fname, str(e)))

    if not frames:
        sys.exit("Error: All CSV files are empty.")

    # --- File Overview ---
    print("=" * 70)
    print("DATA DISCOVERY REPORT")
    print("=" * 70)
    print(f"\nSource directory: {data_dir}")
    print(f"Total CSV files:  {len(csv_files)}")
    print(f"  Non-empty:      {len(frames)}")
    print(f"  Empty:          {len(empty_files)}")

    if empty_files:
        print(f"\nEmpty files: {', '.join(empty_files)}")

    if skipped_files:
        print(f"\nSkipped (parse errors): {len(skipped_files)}")
        for fname, err in skipped_files:
            print(f"  {fname}: {err}")

    # --- File-level summary ---
    print("\n" + "-" * 70)
    print("FILE SUMMARY")
    print("-" * 70)
    print(f"{'File':<45} {'Rows':>7} {'Labels'}")
    print("-" * 70)
    for fname, df in frames:
        labels = ", ".join(sorted(df["label"].dropna().unique())) if "label" in df.columns else "N/A"
        print(f"{fname:<45} {len(df):>7} {labels}")

    # --- Combined dataset ---
    df_all = pd.concat([df for _, df in frames], ignore_index=True)
    print("\n" + "-" * 70)
    print("COMBINED DATASET")
    print("-" * 70)
    print(f"Total rows:    {len(df_all)}")
    print(f"Total columns: {len(df_all.columns)}")
    print(f"Columns:       {list(df_all.columns)}")

    # --- Column types and missing values ---
    print("\n" + "-" * 70)
    print("COLUMN DETAILS")
    print("-" * 70)
    print(f"{'Column':<30} {'Dtype':<12} {'Non-Null':>10} {'Missing':>10} {'Missing%':>10}")
    print("-" * 70)
    for col in df_all.columns:
        non_null = df_all[col].notna().sum()
        missing = df_all[col].isna().sum()
        pct = 100.0 * missing / len(df_all) if len(df_all) > 0 else 0
        print(f"{col:<30} {str(df_all[col].dtype):<12} {non_null:>10} {missing:>10} {pct:>9.1f}%")

    # --- Numeric statistics ---
    numeric_cols = df_all.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        print("\n" + "-" * 70)
        print("NUMERIC STATISTICS")
        print("-" * 70)
        stats = df_all[numeric_cols].describe().T
        stats["missing"] = df_all[numeric_cols].isna().sum()
        print(stats.to_string())

    # --- Label distribution ---
    if "label" in df_all.columns:
        print("\n" + "-" * 70)
        print("LABEL DISTRIBUTION")
        print("-" * 70)
        counts = df_all["label"].value_counts(dropna=False)
        for label, count in counts.items():
            pct = 100.0 * count / len(df_all)
            print(f"  {str(label):<25} {count:>7} ({pct:5.1f}%)")

    # --- Unique values for low-cardinality columns ---
    print("\n" + "-" * 70)
    print("UNIQUE VALUES PER COLUMN")
    print("-" * 70)
    for col in df_all.columns:
        n_unique = df_all[col].nunique()
        sample = ""
        if n_unique <= 10:
            vals = sorted(df_all[col].dropna().unique(), key=str)
            sample = f"  ->  {vals}"
        print(f"  {col:<30} {n_unique:>7} unique{sample}")

    print("\n" + "=" * 70)
    print("END OF REPORT")
    print("=" * 70)
