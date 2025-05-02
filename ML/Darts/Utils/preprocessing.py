from darts.utils import missing_values
from darts.models import KalmanFilter
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
import pandas as pd
import numpy as np
from darts.utils.missing_values import fill_missing_values


def handle_missing_values(timeseries):
    ratio = missing_values.missing_values_ratio(timeseries)
    filled_series = missing_values.fill_missing_values(timeseries)
    return (filled_series, ratio)


def handle_negative_values(timeseries: TimeSeries):
    """Removes entries where the values are zero"""
    mask = timeseries.values().flatten() > 0
    filtered_series = (
        timeseries.drop_before(timeseries.time_index[mask][0]) if mask.any() else None
    )
    return filtered_series


def denoiser(timeseries):
    kf = KalmanFilter(dim_x=1)
    kf.fit(timeseries)
    return kf.filter(timeseries)


def scaler(timeseries: TimeSeries, scaler_instance: Scaler) -> TimeSeries:
    scaler = scaler_instance()
    transformer = Scaler(scaler)
    scaled = transformer.fit_transform(timeseries)
    return scaled


def remove_outliers(series: TimeSeries, outlier_thresh):
    threshold = outlier_thresh
    values = series.values().squeeze()
    cleaned_values = np.where(values > threshold, np.nan, values)
    series_with_nans = series.with_values(cleaned_values)
    interpolated_series = fill_missing_values(series_with_nans, method="linear")

    return interpolated_series


def run_transformer_pipeline(
    timeseries: TimeSeries,
    scale=True,
    scaler_instance=MinMaxScaler,
    resample="h",
    outlier_thresh=3000,
) -> tuple[TimeSeries, float]:
    """Preprocessing pipeline which handles missing values, denoises and scales the timeseries"""
    if resample is not None:
        timeseries.resample(resample)
    timeseries = handle_negative_values(timeseries)
    timeseries = remove_outliers(timeseries, outlier_thresh)
    timeseries, missing_values_ratio = handle_missing_values(timeseries)
    print("Removed missing values")
    print(timeseries.head())
    if scale and scaler_instance is not None:
        print(f"Scaling using {scaler_instance}")
        timeseries = scaler(timeseries=timeseries, scaler_instance=scaler_instance)
    else:
        print("Did not scale data")
    return (timeseries, missing_values_ratio)


def load_data(data: str | list[float, int], granularity=None):
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
            data_no_decimals = [[int(timestamp), value] for timestamp, value in data]
            df = pd.DataFrame(
                data_no_decimals, columns=["timestamp", "value"]
            )  # For json with unix epoch time
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            df = pd.DataFrame(
                data, columns=["timestamp", "value"]
            )  # For json with format YYYY-MM-DD HH:MM
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    ts = TimeSeries.from_dataframe(
        df, time_col=df.columns[0], value_cols=df.columns[1:].tolist(), freq=granularity
    )
    return ts


def load_json_data(json_data):
    return TimeSeries.from_json(json_data)
