import pandas as pd
import re

# Initialize a list to hold date-time entries
date_times = []

# Read the log file line by line and extract the date and time
with open("haproxy.log-anonymized", "r") as file:
    for line in file:
        # Use regex to extract the date-time part (e.g., 22/Jan/2019:03:56:14)
        match = re.search(r"\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})", line)
        if match:
            # Append the extracted date-time to the list
            date_times.append(match.group(1))

# Debugging: check the length of date_times
print(f"Number of extracted date-times: {len(date_times)}")

# Check if date_times is empty
if not date_times:
    raise ValueError("No date-time entries were found in the log file.")

# Ensure 'date_times' is a list of strings
print(f"First few date-time entries: {date_times[:5]}")

# Create a DataFrame from the list of date-times
df = pd.DataFrame(date_times, columns=["date_time"])

# Convert the 'date_time' column to a datetime object for easier handling
df["date_time"] = pd.to_datetime(df["date_time"], format="%d/%b/%Y:%H:%M:%S")

# Round the date-time to the nearest minute
df["date_time_minute"] = df["date_time"].dt.floor("T")

# Count occurrences of each unique date-time (to the minute)
date_time_counts = df["date_time_minute"].value_counts().reset_index()
date_time_counts.columns = ["date_time", "count"]

# Ensure the 'date_time' column is in datetime format
date_time_counts["date_time"] = pd.to_datetime(date_time_counts["date_time"])

# Create a complete range from the minimum to the maximum date-time in the data (per minute)
date_range = pd.date_range(
    start=date_time_counts["date_time"].min(),
    end=date_time_counts["date_time"].max(),
    freq="T",  # 'T' is for minute frequency
)

# Convert the date_range to a DataFrame and set count to 0 initially
all_dates_df = pd.DataFrame(date_range, columns=["date_time"])
all_dates_df["count"] = 0

# Merge the original counts with the full date range
full_date_time_counts = all_dates_df.merge(date_time_counts, on="date_time", how="left")

# Replace NaN values in 'count_y' with 0 and drop the extra count_x column
full_date_time_counts["count"] = full_date_time_counts["count_y"].fillna(0).astype(int)
full_date_time_counts = full_date_time_counts[["date_time", "count"]]

# Save to a CSV file
full_date_time_counts.to_csv("agg_minute.csv", index=False)

print("Date-time counts with filled missing dates saved to 'agg_minute.csv'.")

