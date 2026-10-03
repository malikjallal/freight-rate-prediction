"""Inspect the supplied freight data without modifying the source files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def inspect(frame: pd.DataFrame, name: str) -> None:
    print(f"\n{'=' * 60}\n{name}: {len(frame):,} rows")
    print("Columns:", ", ".join(frame.columns))
    missing = frame.isna().sum()
    print("Missing values:")
    print(missing[missing > 0].to_string() if missing.any() else "None")
    print("Duplicate rows:", int(frame.duplicated().sum()))
    if "load_id" in frame:
        print("Duplicate load IDs:", int(frame.load_id.duplicated().sum()))
    if "date" in frame:
        dates = pd.to_datetime(frame.date, errors="coerce")
        print("Invalid dates:", int(dates.isna().sum()))
        print("Date range:", dates.min(), "to", dates.max())
        print("Rows by month:")
        print(dates.dt.to_period("M").value_counts().sort_index().to_string())
    for column in ["distance", "weight", "market_index", "quote_signal", "posted_rate"]:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        print(f"{column}: non-numeric={int((frame[column].notna() & values.isna()).sum())}, "
              f"infinite={int(np.isinf(values).sum())}, "
              f"non-positive={int((values <= 0).sum())}")
    if "equipment" in frame:
        print("Equipment counts:")
        print(frame.equipment.value_counts(dropna=False).to_string())
    if "posted_rate" in frame:
        print("Target summary:")
        print(frame.posted_rate.describe(percentiles=[.01, .5, .95, .99]).round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    names = ["train-test.csv", "validation.csv", "validation-predictions-template.csv",
             "december-chart-inputs.csv"]
    paths = [args.data_dir / name for name in names]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"File not found: {path}. Use --data-dir to specify the ML folder.")
    train, validation, template, december = [pd.read_csv(path) for path in paths]
    inspect(train, names[0])
    inspect(validation, names[1])
    print("\nFINAL INPUT CHECKS")
    if validation.load_id.isna().any() or validation.load_id.duplicated().any():
        raise SystemExit("Validation IDs must be present and unique.")
    if template.load_id.isna().any() or template.load_id.duplicated().any():
        raise SystemExit("Template IDs must be present and unique.")
    if set(validation.load_id) != set(template.load_id):
        raise SystemExit("Template IDs do not match the validation IDs.")
    print("Template IDs match validation:", len(template))
    print("Overlapping train/validation IDs:", len(set(train.load_id) & set(validation.load_id)))
    print("December input rows:", len(december))
    available = set(december.columns) - {"predicted_rate"}
    required = set(train.columns) - {"load_id", "posted_rate"}
    print("Features absent from December inputs:", ", ".join(sorted(required - available)))
    print("\nInspection complete. No source files were changed.")


if __name__ == "__main__":
    main()
