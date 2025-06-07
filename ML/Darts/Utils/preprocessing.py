from datetime import datetime
from darts.utils import missing_values
from darts.models import KalmanFilter
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from darts import TimeSeries
import pandas as pd
import numpy as np
from darts.utils.missing_values import fill_missing_values

from Database.Entities.Historical import Historical

def handle_missing_values(timeseries):
    ratio = missing_values.missing_values_ratio(timeseries)
    filled_series = missing_values.fill_missing_values(timeseries)
    return (filled_series, ratio)


def handle_negative_values(timeseries: TimeSeries) -> TimeSeries:
    """Removes entries where the values are zero"""
    mask = timeseries.values().flatten() > 0
    if not mask.any():
        raise ValueError("Empty series")
    filtered = timeseries.drop_before(timeseries.time_index[mask][0])
    return filtered


def denoiser(timeseries):
    kf = KalmanFilter(dim_x=1)
    kf.fit(timeseries)
    return kf.filter(timeseries)

def scaler(timeseries: TimeSeries) -> tuple[TimeSeries, Scaler]:
    transformer = Scaler(MinMaxScaler(feature_range=(0, 1)))
    scaled = transformer.fit_transform(timeseries)
    if not isinstance(scaled, TimeSeries):
        raise ValueError("Error, wrong timeseries type in scaler type")
    return (scaled, transformer)

def remove_outliers(series: TimeSeries | None, outlier_thresh):
    if series is None:
        raise ValueError("TimeSeries is None.")
    threshold = outlier_thresh
    values = series.values().squeeze()
    cleaned_values = np.where(values > threshold, np.nan, values)
    series_with_nans = series.with_values(cleaned_values)
    interpolated_series = fill_missing_values(series_with_nans, method="linear")

    return interpolated_series


def run_transformer_pipeline(
    timeseries: TimeSeries,
    resample="h",
    outlier_thresh=3000,
) -> tuple[TimeSeries, float, Scaler]:
    """Preprocessing pipeline which handles missing values, denoises and scales the timeseries"""
    if resample is not None:
        timeseries.resample(resample)
    non_negative_timeseries = handle_negative_values(timeseries)
    without_outliers_timeseries = remove_outliers(non_negative_timeseries, outlier_thresh)
    handled_missing_timeseries, missing_values_ratio = handle_missing_values(without_outliers_timeseries)
    print("Removed missing values")

    print(f"Scaling data")
    final_timeseries, transformer = scaler(handled_missing_timeseries)

    return (final_timeseries, missing_values_ratio, transformer)

def unscaling_pipeline(timeseries: TimeSeries, scaler: Scaler, period: pd.Timedelta) -> TimeSeries:
    inverse_timeseries = scaler.inverse_transform(timeseries)
    if not isinstance(inverse_timeseries, TimeSeries):
        raise ValueError("TimeSeries is not a TimeSeries")
    resampled_timeseries = inverse_timeseries.resample(freq=f"{int(period.total_seconds())}s")
    cleaned_timeseries = TimeSeries.from_dataframe(resampled_timeseries.pd_dataframe().dropna())
    return cleaned_timeseries

def load_data(data: str | list[tuple[float, int]], granularity=None):
    """
    Args:
        data_path (str): Path to csv
        granularity (str): The interval between each timestamp. Must be one of these: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """
    if isinstance(data, str) and data.endswith(".csv"):  # For CSV
        df = pd.read_csv(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        if isinstance(data[0][0], (int, float)):
            data_no_decimals = [(int(timestamp), value) for timestamp, value in data]
            df = pd.DataFrame(
                data_no_decimals, columns=pd.Index(["timestamp", "value"])
            )  # For json with unix epoch time
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            df = pd.DataFrame(
                data, columns=pd.Index(["timestamp", "value"])
            )  # For json with format YYYY-MM-DD HH:MM
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts = TimeSeries.from_dataframe(
        df, time_col="timestamp", value_cols=df.columns[1:].tolist(), freq=granularity
    )
    return ts

def load_historical_data(data:Historical, horizon:pd.Timedelta) -> TimeSeries:
    series = {
        "timestamp": [datetime.fromtimestamp(value[0]) for value in data.data["data"]["result"][0]["values"]],
        "value": [float(value[1]) for value in data.data["data"]["result"][0]["values"]]
    }
    df = pd.DataFrame(series)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    final_series = TimeSeries.from_dataframe(df, time_col='timestamp', value_cols='value', fill_missing_dates=True, freq=f"{int(horizon.total_seconds()/60)}s").astype("float32")
    return final_series


def load_json_data(json_data):
    tuning_data = json_data["tuning_data"]
    df = pd.DataFrame(tuning_data)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    series = TimeSeries.from_dataframe(df, time_col='timestamp', value_cols='value')

    if series is None:
        raise ValueError("TimeSeries did not load properly.")
    
    return series
