# Freight rate prediction

Predict `posted_rate` for the assessment's 12,000 validation loads and the fixed December route.

## Setup

The recorded run used Windows, Python 3.14.3 and scikit-learn 1.9.1. The requirements pin the main packages from that run.

Keep these supplied files in the project root:

- `train-test.csv`
- `validation.csv`
- `validation-predictions-template.csv`
- `december-chart-inputs.csv`
- `score.py`

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe explore.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe predict.py
.\.venv\Scripts\python.exe score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

On macOS/Linux, create a virtual environment with `python3 -m venv .venv`, then use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## Validation and model

January-August contains 38,477 training rows. September's 4,670 rows select the feature set by MAE in dollars. After selection, the chosen model is refitted on January-September and evaluated on October's 4,853 rows. The external November-December validation loads have no target values and are not used for selection.

Three fixed feature sets are compared using HistGradientBoostingRegressor: core, core plus coordinates and market index, and the extended set plus quote signal. A median rate-per-mile baseline by equipment type is also evaluated.

The core set won on September. It uses pickup, delivery, equipment, distance, weight, weekday and two cyclic day-of-year features. It also covers all inputs supplied for the fixed December chart.

Weights at or below zero are treated as missing. Numeric medians and categorical encodings are fitted inside each training pipeline. The model uses absolute-error loss on rate per mile, weighted by distance. Predictions are multiplied by distance to recover dollars. Target outliers are retained.

## Recorded results

MAE and RMSE are in dollars. Values below are rounded to three decimals from the Windows run.

| September model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Core | 108.136 | 617.820 | 0.836 |
| Extended + quote | 114.640 | 619.104 | 0.835 |
| Extended | 121.288 | 621.333 | 0.834 |
| Equipment median | 227.323 | 657.066 | 0.814 |

| October model | MAE | RMSE | R2 |
| --- | ---: | ---: | ---: |
| Selected core | 117.935 | 650.518 | 0.819 |
| Equipment median | 231.305 | 683.865 | 0.800 |

October MAE improves by about 49% against the baseline. RMSE remains much larger than MAE, so large errors remain. R2 is not a percentage accuracy score.

## Outputs

- `results/`: September and October metrics, October predictions, selected model name and fitted model artifact.
- `validation_predictions.csv`: exactly `load_id,predicted_rate`, in template order.
- `december_predictions.csv`: the original December inputs plus predictions.
- `scorer_results/candidate_december.png`: chart created by the supplied, unchanged scorer.
- `Freight_Rate_Report.pdf`: approach, results and December chart.

The scorer checks file structure, IDs and positive rates. It does not measure prediction accuracy; Spotter calculates the final metrics after submission. Seasonal features are learned from January-October only, so the December chart is a forecast, not evidence of December accuracy.

The original input CSVs are preserved. Each script supports `--data-dir`; training and prediction also support `--output-dir`.
