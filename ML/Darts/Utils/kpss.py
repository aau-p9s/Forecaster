from darts.utils.statistics import stationarity_test_kpss, extract_trend_and_seasonality
import pandas as pd
from darts import TimeSeries
import matplotlib.pyplot as plt

df = pd.read_csv("agg_minute.csv")  # Update the path to your dataset
df["time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")
series = TimeSeries.from_dataframe(
    df, time_col="time", value_cols="count", freq="min"
)

statistic, p_value, lags, crit = stationarity_test_kpss(series)

print(f"KPSS Statistic - Minute: {statistic}. P-value: {p_value}. Lags: {lags}. Crit: {crit}\n")

if p_value < 0.05:
    print("The time series is likely non-stationary.\n")
else:
    print("The time series is likely stationary.\n")
# Plot and save the TimeSeries
plt.figure(figsize=(12, 6))
series.plot()
plt.title("TimeSeries Plot")
plt.savefig("timeseries_minutely_plot.png")
plt.close()

df = pd.read_csv("agg_daily.csv")  # Update the path to your dataset
df["time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d")
series = TimeSeries.from_dataframe(
    df, time_col="time", value_cols="count", freq="d"
)

statistic, p_value, lags, crit = stationarity_test_kpss(series)

print(f"KPSS Statistic - Daily: {statistic}. P-value: {p_value}. Lags: {lags}. Crit: {crit}\n")

if p_value < 0.05:
    print("The time series is likely non-stationary.\n")
else:
    print("The time series is likely stationary.\n")


df = pd.read_csv("agg_second.csv")  # Update the path to your dataset
df["time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")
series = TimeSeries.from_dataframe(
    df, time_col="time", value_cols="count", freq="s"
)

statistic, p_value, lags, crit = stationarity_test_kpss(series)

print(f"KPSS Statistic - Second: {statistic}. P-value: {p_value}. Lags: {lags}. Crit: {crit}\n")

if p_value < 0.05:
    print("The time series is likely non-stationary.\n")
else:
    print("The time series is likely stationary.\n")
