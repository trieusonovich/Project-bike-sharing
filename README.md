# Project-bike-sharing

Predicting the number of bike-sharing users per hour from weather conditions and calendar information, using a **Random Forest** model built inside a full scikit-learn preprocessing pipeline and tuned with **GridSearchCV**.

## Dataset

- `bike-sharing.csv`: 17,544 hourly records from 01/01/2011 to 31/12/2012 (2 years).
- Target: `users` (number of rentals per hour). Mean ≈ 188, max = 977.
- Features:
  - Weather: `temp`, `atemp` (feels-like temperature), `hum` (humidity), `windspeed`, `weather` (clear / mist / rain)
  - Calendar: `holiday`, `workingday`, `month`, `hour`, `weekday`
- Data quality: no duplicates; only 2 missing values (in `weather`).
- `report_bike_sharing.html`: automated EDA report (ydata-profiling).

## Approach

1. **EDA:** profiling report to inspect distributions, correlations and missing values.
2. **Time-ordered split:** first 80% of the timeline for training, last 20% (from 07/08/2012) for testing, with `shuffle=False`.
3. **Preprocessing with `ColumnTransformer`** (fit on training data only, so no leakage into the test set):

   | Column(s) | Transformation |
   |---|---|
   | `atemp` | QuantileTransformer (normal) → StandardScaler |
   | `temp`, `hum` | Median imputation → MinMaxScaler |
   | `windspeed` | Quantile binning into 3 ordinal groups |
   | `weather` | Mode imputation → One-hot encoding |
   | `month`, `hour`, `weekday` | Cyclical (sin/cos) encoding via `feature_engine.CyclicalFeatures` |
   | `holiday`, `workingday` | Passed through |

4. **Model:** `RandomForestRegressor` inside a single `Pipeline`.
5. **Hyperparameter tuning:** `GridSearchCV` (scoring = R², 4-fold) over `n_estimators`, `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
6. **Evaluation:** MAE, MSE, R² on the held-out test period.

## Results

| Metric | Test set |
|---|---|
| MAE | ≈ 88 users/hour |
| MSE | ≈ 15,700 |
| R² | ≈ 0.68 |

For reference, predicting the training mean gives R² < 0 on the same test set.

> Replace these numbers with the output of your own `GridSearchCV` run (`best_params_` and the test metrics printed by the script).

## Key Finding: Demand Grew Over Time

The average number of users in the training period is ~173/hour, while in the test period it is ~246/hour. Bike-sharing usage grew from 2011 to 2012, and the original features contain no information about the year, so a tree-based model cannot anticipate this growth.

Adding a simple `year` feature (derived from the timestamp, which is always known at prediction time) raised test R² from about **0.68 to 0.85** and cut MAE from about 88 to 57 in a quick experiment with the same pipeline.

## Project Structure

```
├── bike-sharing.py                 # Pipeline, tuning and evaluation
├── bike-sharing.csv                # Dataset
├── report_bike_sharing.html        # EDA report
└── README.md
```

## How to Run

```bash
pip install pandas scikit-learn matplotlib feature-engine
python bike-sharing.py
```

Note: the full grid contains 288 parameter combinations × 4 folds and `criterion="absolute_error"` is very slow, so tuning can take a long time. Reduce the grid for a quick test.

## Limitations and Next Steps

- Add `year` (or a time index) as a feature to handle the growth trend.
- Use `TimeSeriesSplit` instead of plain 4-fold CV, so validation folds never precede the training data.
- Scaling and binning are not required for tree-based models; compare against a simpler pipeline, and against Gradient Boosting / XGBoost / LightGBM.
- Add feature importance and residual analysis (errors by hour, weekday vs. weekend, weather).

## Tech Stack

Python · pandas · scikit-learn · feature-engine · Matplotlib · ydata-profiling

## Author

Nguyen Dinh Trieu

Gmail: trieu31072004@gmail.com
