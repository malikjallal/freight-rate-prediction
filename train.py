"""Compare freight-rate models using chronological development splits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits


CATEGORIES = ["pickup", "delivery", "equipment"]
CORE = ["distance", "weight", "weekday", "day_of_year_sin", "day_of_year_cos"]
EXTRA = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index"]
VARIANTS = {
    "core": CORE,
    "extended": CORE + EXTRA,
    "extended_quote": CORE + EXTRA + ["quote_signal"],
}


def features(data):
    result = data.copy()
    dates = pd.to_datetime(result["date"], errors="raise")
    result["weekday"] = dates.dt.dayofweek
    angle = 2 * np.pi * (dates.dt.dayofyear - 1) / 365.25
    result["day_of_year_sin"] = np.sin(angle)
    result["day_of_year_cos"] = np.cos(angle)
    for column in set(CORE + EXTRA + ["quote_signal"]) - {
        "weekday", "day_of_year_sin", "day_of_year_cos"
    }:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    result.loc[result["weight"] <= 0, "weight"] = np.nan
    for column in CATEGORIES:
        result[column] = result[column].fillna("Unknown").astype(str)
    return result


def make_model(numeric):
    preprocessing = ColumnTransformer([
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         CATEGORIES),
        ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
    ])
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=20,
        min_samples_leaf=40,
        l2_regularization=2.0,
        categorical_features=[0, 1, 2],
        early_stopping=False,
        random_state=42,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    }


def fit_model(model, data):
    rate_per_mile = data["posted_rate"] / data["distance"]
    # Weighting by distance makes absolute error in rate-per-mile match dollar error.
    model.fit(data, rate_per_mile, model__sample_weight=data["distance"])
    return model


def predict(model, data):
    result = model.predict(data) * data["distance"].to_numpy()
    if not np.isfinite(result).all():
        raise ValueError("The model produced non-finite predictions.")
    return np.maximum(result, 0.01)


def baseline(train, evaluation):
    rates = (train["posted_rate"] / train["distance"]).groupby(train["equipment"]).median()
    fallback = (train["posted_rate"] / train["distance"]).median()
    return evaluation["equipment"].map(rates).fillna(fallback) * evaluation["distance"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.data_dir / "train-test.csv"
    output = args.output_dir or args.data_dir / "results"
    data = pd.read_csv(source)
    dates = pd.to_datetime(data["date"], errors="raise")
    data = features(data)
    if data["distance"].isna().any() or (data["distance"] <= 0).any():
        raise ValueError("Training distances must be finite and positive.")
    if not np.isfinite(data["posted_rate"]).all() or (data["posted_rate"] <= 0).any():
        raise ValueError("Training target must be finite and positive.")
    train = data.loc[dates < "2025-09-01"].copy()
    selection = data.loc[(dates >= "2025-09-01") & (dates < "2025-10-01")].copy()
    test = data.loc[dates >= "2025-10-01"].copy()
    if min(len(train), len(selection), len(test)) == 0:
        raise ValueError("Each chronological split must contain data.")
    print(f"Training: January-August ({len(train):,} rows)")
    print(f"Model selection: September ({len(selection):,} rows)")
    print(f"Holdout: October ({len(test):,} rows)")
    print("Training medians are fitted inside each pipeline.")
    print("Model selection metric: September MAE in dollars.\n")
    rows = [{"model": "equipment_median", **metrics(selection.posted_rate, baseline(train, selection))}]
    with threadpool_limits(limits=4):
        for name, numeric in VARIANTS.items():
            print(f"Training {name}...", flush=True)
            model = fit_model(make_model(numeric), train)
            rows.append({"model": name, **metrics(selection.posted_rate, predict(model, selection))})
        summary = pd.DataFrame(rows).sort_values("MAE")
        print("\nSeptember results:")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        winner = summary.iloc[0]["model"]
        development = data.loc[dates < "2025-10-01"].copy()
        if winner == "equipment_median":
            predictions = baseline(development, test)
        else:
            chosen = fit_model(make_model(VARIANTS[winner]), development)
            predictions = predict(chosen, test)
    test_rows = [
        {"model": "equipment_median", **metrics(test.posted_rate, baseline(development, test))},
        {"model": winner, **metrics(test.posted_rate, predictions)},
    ]
    holdout = pd.DataFrame(test_rows).drop_duplicates("model")
    print(f"\nSelected model: {winner}")
    print("October results (model selected before viewing these scores):")
    print(holdout.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "september_metrics.csv", index=False)
    holdout.to_csv(output / "october_metrics.csv", index=False)
    pd.DataFrame({"load_id": test.load_id, "actual_rate": test.posted_rate,
                  "predicted_rate": np.asarray(predictions)}).to_csv(
                      output / "october_predictions.csv", index=False)
    (output / "selected_model.txt").write_text(winner + "\n", encoding="utf-8")
    print(f"\nResults saved to: {output}")
    print("The external validation data has not been used for model selection.")


if __name__ == "__main__":
    main()
