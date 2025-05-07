from darts.utils import missing_values
from darts.models import KalmanFilter
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
import pandas as pd

from enum import Enum

class ScalerType(str, Enum):
    MINMAX = "minmax"
    STANDARD = "standard"
    NONE = "none"

def build_scaler(scaler_type: ScalerType) -> Scaler:
    if scaler_type == ScalerType.MINMAX:
        scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
    elif scaler_type == ScalerType.STANDARD:
        scaler = Scaler(StandardScaler())
    else:
        return None
    scaler._fit_called = True  # trick if using saved params manually
    return scaler

def handle_missing_values(timeseries):
    ratio = missing_values.missing_values_ratio(timeseries)
    filled_series = missing_values.fill_missing_values(timeseries)
    return (filled_series, ratio)

def denoiser(timeseries):
    kf = KalmanFilter(dim_x=1)
    kf.fit(timeseries)
    return kf.filter(timeseries)

def scaler(timeseries: TimeSeries, scaler_type: Scaler) -> TimeSeries:
    transformer = build_scaler(scaler_type)
    scaled = transformer.fit_transform(timeseries)
    return scaled

def run_transformer_pipeline(timeseries: TimeSeries):
    """Preprocessing pipeline which handles missing values, denoises and scales the timeseries"""
    timeseries, missing_values_ratio = handle_missing_values(timeseries)
    timeseries = scaler(timeseries)
    return (timeseries, missing_values_ratio)

def load_data(data: str | list[float, int], granularity=None):
        """
        Args:
            data_path (str): Path to csv
            granularity (str): The interval between each timestamp. Must be one of these: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        """
        if isinstance(data, str) and data.endswith(".csv"): # For CSV
            df = pd.read_csv(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            if isinstance(data[0][0], (int, float)):
                data_no_decimals = [[int(timestamp), value] for timestamp, value in data]
                df = pd.DataFrame(data_no_decimals, columns=["timestamp", "value"]) # For json with unix epoch time
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df = pd.DataFrame(data, columns=["timestamp", "value"]) # For json with format YYYY-MM-DD HH:MM
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        ts = TimeSeries.from_dataframe(df, time_col=df.columns[0], value_cols=df.columns[1:].tolist(), freq=granularity)
        return ts

def load_json_data(json_data):
    return TimeSeries.from_json(json_data)