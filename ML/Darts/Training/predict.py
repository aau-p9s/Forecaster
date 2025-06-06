from multiprocessing import Pool, cpu_count
import traceback
from darts.dataprocessing.transformers.scaler import Scaler
from darts.metrics.metrics import rmse
from darts.timeseries import TimeSeries
from pandas import Timedelta
from Database.Models.Forecast import Forecast
from Database.Models.Model import Model
import pandas as pd

from ML.Darts.Utils.preprocessing import unscaling_pipeline
from ML.Darts.Utils.timeout import timeout

def predict(model: Model, series: TimeSeries, scaler: Scaler, period: Timedelta, horizon: Timedelta):
    try:
        print(f"Predicting for period: {period} seconds", flush=True)

        # scale to seconds, and scale slightly more to adjust for the validation series during training
        trained_frequency = model.get_trained_frequency(default=pd.to_timedelta("1m"))
        print(f"Trained frequency: {trained_frequency}")
        prediction_period = int(period / trained_frequency) * 2
        forecast = timeout(model.model.predict, prediction_period)

        print("Created forecast", flush=True)
        forecast_rmse = rmse(series, forecast)
        print("Calculated RMSE", flush=True)
        unscaled_forecast = unscaling_pipeline(forecast, scaler, horizon)
        print(f"Pipeline output: length {len(unscaled_forecast)} for period {unscaled_forecast.time_index[0]} to {unscaled_forecast.time_index[-1]}", flush=True)
        return Forecast(model.modelId, unscaled_forecast, forecast_rmse)
    except Exception as e:
        traceback.print_exc()
        print(f"Model {model.name} failed, continuing to next model: {e}", flush=True)


    return timeout(model.model.predict, int(period.total_seconds()))

def predict_all(models: list[Model], series: TimeSeries, scaler: Scaler, period: Timedelta, horizon: Timedelta):
    with Pool(int(cpu_count()/2)) as pool:
        predictions = pool.starmap(predict, [(model, series, scaler, period, horizon) for model in models])
    successful_predictions = []
    for prediction in predictions:
        if prediction is not None:
            successful_predictions.append(prediction)
    return successful_predictions
