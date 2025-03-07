from darts.utils import missing_values
from darts.models import KalmanFilter
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
import matplotlib.pyplot as plt

data = AirPassengersDataset().load()

def handle_missing_values(timeseries):
    ratio = missing_values.missing_values_ratio(timeseries)
    filled_series = missing_values.fill_missing_values(timeseries)
    return (filled_series, ratio)

def denoiser(timeseries):
    kf = KalmanFilter(dim_x=1)
    kf.fit(timeseries)
    return kf.filter(timeseries)

def scaler(timeseries):
    scaler = Scaler()
    scaled = scaler.fit_transform(timeseries)
    return scaled

def run_transformer_pipeline(timeseries):
    """Preprocessing pipeline which handles missing values, denoises and scales the timeseries"""
    timeseries = handle_missing_values(timeseries)[0]
    timeseries = scaler(timeseries)
    return timeseries