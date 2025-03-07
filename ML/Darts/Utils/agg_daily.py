import pandas as pd

# Load the existing CSV that contains minute-level data
df = pd.read_csv("agg_minute.csv")
print(df.columns)
# Ensure 'date_time' is a datetime object
df["date_time"] = pd.to_datetime(df["date_time"])

# Group by the day and sum the counts for that day
df["date_time"] = df["date_time"].dt.floor("D")  # Round to the start of the day

# Group by the day and sum the counts
daily_counts = df.groupby("date_time")["count"].sum().reset_index()

# Create a complete range from the minimum to the maximum date-time in the data (per day)
date_range = pd.date_range(
    start=daily_counts["date_time"].min(),
    end=daily_counts["date_time"].max(),
    freq="D",  # 'D' is for daily frequency
)

# Convert the date_range to a DataFrame and set count to 0 initially
all_dates_df = pd.DataFrame(date_range, columns=["date_time"])
all_dates_df["count"] = 0
print(all_dates_df.head())

# Merge the original daily counts with the full date range
full_daily_counts = all_dates_df.merge(daily_counts, on='date_time', how="left")

# Replace NaN values in 'count_y' with 0
full_daily_counts["count"] = full_daily_counts["count_y"].fillna(0).astype(int)
full_daily_counts = full_daily_counts[["date_time", "count"]]

# Save the aggregated daily data to a CSV
full_daily_counts.to_csv("agg_daily.csv", index=False)

print("Daily aggregated counts saved to 'agg_daily.csv'.")
