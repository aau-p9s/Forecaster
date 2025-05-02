from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
import numpy as np
import matplotlib.pyplot as plt

# Load the series
series = TimeSeries.from_csv(
    "../../Datasets/combined/combined_output.csv",
    time_col="timestamp",
    value_cols="value",
    fill_missing_dates=True,
    freq="min",  # 'T' stands for minutely frequency
)

# Outlier removal: replace high values with NaN
threshold = 3000
values = series.values().squeeze()
cleaned_values = np.where(values > threshold, np.nan, values)
series_with_nans = series.with_values(cleaned_values)

interpolated_series = fill_missing_values(series_with_nans, method="linear")

interpolated_series.to_csv("interpolated_outliers_removed_full_dataset.csv")
# Plot and save
plt.figure()
series.plot(label="Original")
interpolated_series.plot(label="Cleaned")
plt.legend()
plt.title("Outlier Removal with Interpolation")
plt.savefig("cleaned_series.png")
