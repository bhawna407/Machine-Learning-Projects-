# encoding: utf-8
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# 1. Convert the 'fake' zeros back to NaN so interpolate can see them
df['y'] = df['y'].replace(0, np.nan)

# 2. Fill the gaps with a straight line
df['y'] = df['y'].interpolate(method='linear')

# 3. If there are still NaNs at the very beginning, fill them
df['y'] = df['y'].fillna(method='bfill')


# 1. Load & prepare
df = pd.read_csv(r"C:\Users\PC\Downloads\P04 DAY 2 TASK FORECASTING SALES LLM\OLD\clean_daily_sales.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

print(f"Date range : {df['ds'].min().date()} to {df['ds'].max().date()}")
print(f"Total rows : {len(df)}")

# 2. Train/test split (before June 2018 = train, June 2018 onwards = test)
split_date = pd.Timestamp("2018-06-01")
train = df[df["ds"] < split_date].copy()
test  = df[df["ds"] >= split_date].copy()

print(f"\nTrain rows : {len(train)}  ({train['ds'].min().date()} to {train['ds'].max().date()})")
print(f"Test  rows : {len(test)}   ({test['ds'].min().date()} to {test['ds'].max().date()})")

# Helper: predict on a set of specific dates
def predict_on_dates(model, target_dates):
    """Ask Prophet to predict exactly on the given dates."""
    future = pd.DataFrame({"ds": target_dates})
    forecast = model.predict(future)
    return forecast.set_index("ds")["yhat"].loc[target_dates].values

# 3. Tune changepoint_prior_scale
print("\nTuning changepoint_prior_scale ...")
scales  = [0.01, 0.05, 0.1, 0.3, 0.5]
results = {}

for scale in scales:
    m = Prophet(changepoint_prior_scale=scale, daily_seasonality=False)
    m.fit(train)
    preds = predict_on_dates(m, test["ds"].values)
    mae   = mean_absolute_error(test["y"].values, preds)
    results[scale] = mae
    print(f"  scale={scale:<5}  MAE={mae:,.2f}")

best_scale = min(results, key=results.get)
best_mae   = results[best_scale]
print(f"\nBest changepoint_prior_scale : {best_scale}  MAE = {best_mae:,.2f}")

# 4. Final test evaluation with best scale
model_test = Prophet(changepoint_prior_scale=best_scale, daily_seasonality=False)
model_test.fit(train)
test_preds = predict_on_dates(model_test, test["ds"].values)
final_mae  = mean_absolute_error(test["y"].values, test_preds)
print(f"\nFinal MAE on test set : {final_mae:,.2f}")

# 5. Train on full data and predict next 90 days
print("\nTraining on full data and forecasting next 90 days ...")
model_full    = Prophet(changepoint_prior_scale=best_scale, daily_seasonality=False)
model_full.fit(df)
future_full   = model_full.make_future_dataframe(periods=90)
forecast_full = model_full.predict(future_full)

last_date   = df["ds"].max()
forecast_90 = forecast_full[forecast_full["ds"] > last_date][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
forecast_90 = forecast_90.head(90).reset_index(drop=True)

print(forecast_90.head())
print(f"\nForecast rows : {len(forecast_90)}")
print(f"Forecast from : {forecast_90['ds'].min().date()} to {forecast_90['ds'].max().date()}")

# 6. Save submission file
out_path = r"C:\Users\PC\Downloads\P04 DAY 2 TASK FORECASTING SALES LLM\forecast_results.csv"
forecast_90.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# 7. Summary
print("\n" + "="*50)
print("     DAY 2 SUBMISSION SUMMARY")
print("="*50)
print(f"  Best changepoint_prior_scale : {best_scale}")
print(f"  MAE on test set              : {final_mae:,.2f}")
print(f"  Forecast file                : forecast_results.csv")
print(f"  Columns                      : ds, yhat, yhat_lower, yhat_upper")
print(f"  Forecast period              : next 90 days")
print("="*50)
