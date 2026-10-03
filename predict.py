"""Refit the selected model and create the two submission prediction files."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from train import CATEGORIES, VARIANTS, baseline, features, fit_model, make_model, predict


def require_ids(frame, label):
    if "load_id" not in frame:
        raise ValueError(f"{label} has no load_id column.")
    if frame.load_id.isna().any() or frame.load_id.duplicated().any():
        raise ValueError(f"{label} contains missing or duplicate load IDs.")


def require_distance(frame, label):
    values = frame["distance"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label} distances must be finite and positive.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-file", type=Path)
    args = parser.parse_args()
    source = args.data_dir
    output = args.output_dir or source
    selection_file = args.selection_file or source / "results" / "selected_model.txt"
    if not selection_file.is_file():
        raise SystemExit("Run train.py first to select a model using September data.")
    selected = selection_file.read_text(encoding="utf-8").strip()
    if selected not in VARIANTS and selected != "equipment_median":
        raise ValueError(f"Unknown selected model: {selected}")

    train_raw = pd.read_csv(source / "train-test.csv")
    final_raw = pd.read_csv(source / "validation.csv")
    template = pd.read_csv(source / "validation-predictions-template.csv")
    december_raw = pd.read_csv(source / "december-chart-inputs.csv")
    require_ids(train_raw, "Training data")
    require_ids(final_raw, "Validation data")
    require_ids(template, "Prediction template")
    if len(final_raw) != 12_000 or len(template) != 12_000:
        raise ValueError("Validation data and template must each contain 12,000 rows.")
    if set(final_raw.load_id) != set(template.load_id):
        raise ValueError("Validation and template IDs do not match.")
    if set(train_raw.load_id) & set(final_raw.load_id):
        raise ValueError("Training and validation IDs overlap.")
    if list(december_raw.columns) != [
        "pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"
    ]:
        raise ValueError("Unexpected December input columns.")
    if len(december_raw) != 31:
        raise ValueError("December inputs must contain 31 rows.")
    train = features(train_raw)
    final = features(final_raw)
    december = features(december_raw)
    for frame, label in [(train, "Training"), (final, "Validation"), (december, "December")]:
        require_distance(frame, label)
    if not np.isfinite(train.posted_rate).all() or (train.posted_rate <= 0).any():
        raise ValueError("Training rates must be finite and positive.")
    if selected != "equipment_median":
        required = set(CATEGORIES + VARIANTS[selected])
        for frame, label in [(final, "Validation"), (december, "December")]:
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{label} is missing selected-model features: {sorted(missing)}")

    print(f"Selected model: {selected}")
    print(f"Refitting on all {len(train):,} labeled loads...", flush=True)
    with threadpool_limits(limits=4):
        if selected == "equipment_median":
            final_rates = baseline(train, final).to_numpy()
            december_rates = baseline(train, december).to_numpy()
            artifact = {"model_name": selected,
                        "equipment_rates": (train.posted_rate / train.distance)
                        .groupby(train.equipment).median(),
                        "fallback_rate": (train.posted_rate / train.distance).median()}
        else:
            model = fit_model(make_model(VARIANTS[selected]), train)
            final_rates = predict(model, final)
            december_rates = predict(model, december)
            artifact = {"model_name": selected, "pipeline": model,
                        "target": "posted_rate / distance",
                        "required_features": CATEGORIES + VARIANTS[selected]}

    mapping = pd.Series(final_rates, index=final.load_id)
    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission.load_id.map(mapping)
    december_submission = december_raw.copy()
    december_submission["predicted_rate"] = december_rates
    for rates in [submission.predicted_rate, december_submission.predicted_rate]:
        if not np.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("All predicted rates must be finite and positive.")

    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / "validation_predictions.csv", index=False, float_format="%.4f")
    december_submission.to_csv(output / "december_predictions.csv", index=False, float_format="%.4f")
    model_dir = output / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_dir / "freight_model.joblib")
    print("Created validation_predictions.csv: 12,000 predictions")
    print("Created december_predictions.csv: 31 predictions")
    print("Saved results/freight_model.joblib")
    print("Original input CSV files were not modified.")


if __name__ == "__main__":
    main()
