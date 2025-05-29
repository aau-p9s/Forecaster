import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# 1. Load your data and make sure you have a datetime index or column
files = glob("../../Datasets/ServerRequest*.csv")
df_list = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
df = pd.concat(df_list)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# 2. Compute minute-of-week for each row
#    Monday=0, Sunday=6. Each day has 1 440 minutes.

df = df.reset_index()
df["minute_of_week"] = (
    df["timestamp"].dt.dayofweek * 1440
    + df["timestamp"].dt.hour * 60
    + df["timestamp"].dt.minute
)
df = df.set_index("timestamp")

# 3. Group by minute-of-week and aggregate
#    Here I’ll sum requests, but you can use `.mean()`, `.median()`, etc.
weekly = df.groupby("minute_of_week")["requests"].sum()

# 4. (Optional) Re-index to a proper one-week time axis
#    Create a dummy “week” starting on a Monday at midnight:
start = pd.to_datetime("2025-01-06")  # any Monday works
time_index = pd.date_range(start, periods=7 * 1440, freq="T")
weekly.index = time_index  # now `weekly` is time-indexed over one week

# 5. Access your result
#    `weekly` is now a Series of length 10080 (7×1440), where each timestamp
#    corresponds to the aggregated sum of that minute-of-week across all your data.
print(weekly.head())  # shows Monday 00:00, 00:01, …

weekly.to_csv("../../Datasets/combined/weekly_requests_summary.csv")

plt.figure(figsize=(12, 5))
weekly.plot()
plt.xlabel("Time")
plt.ylabel("Accumulated Requests")
plt.title("Accumulated Requests per minute")
plt.grid(True)
plt.tight_layout()
plt.savefig("combined_by_minute_across_all_days.png")
