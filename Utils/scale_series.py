from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import matplotlib.pyplot as plt
import numpy as np

# Load the series
series = TimeSeries.from_csv(
    "./interpolated_outliers_removed_full_dataset.csv",
    time_col="timestamp",
    value_cols="value",
    fill_missing_dates=True,
    freq="min",
)

series = series.resample("H")

# Scale the series using MinMaxScaler
scaler = Scaler()
scaled_series = scaler.fit_transform(series)

# Save scaled data to CSV
scaled_series.pd_dataframe().to_csv("scaled_series_full.csv")

# Plot and save figure
plt.figure()
scaled_series.plot(label="Scaled Series")
plt.title("MinMax Scaled Time Series")
plt.legend()
plt.savefig("scaled_plot_full.png")
