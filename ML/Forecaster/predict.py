from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from multiprocessing.managers import ValueProxy
import traceback
from darts.dataprocessing.transformers.scaler import Scaler
from darts.metrics.metrics import rmse
from darts.timeseries import TimeSeries
from pandas import Timedelta
import pandas as pd
import numpy as np

from Database.Entities.Forecast import Forecast
from Database.Entities.Model import Model
from ML.Darts.Utils.preprocessing import unscaling_pipeline
from ML.Darts.Utils.split_models import split_models
from ML.Darts.Utils.timeout import timeout

def predict(model: Model, series: TimeSeries, scaler: Scaler, period: Timedelta, horizon: Timedelta, finished: ValueProxy) -> Forecast | None:
    try:
        print(f"Predicting for period: {period} seconds", flush=True)

        # scale to seconds, and scale slightly more to adjust for the validation series during training
        trained_frequency = model.get_trained_frequency(default=pd.to_timedelta("1m"))
        print(f"Trained frequency: {trained_frequency}")
        prediction_period = int(period / trained_frequency) * 2
        forecast = timeout(model.model.predict, prediction_period)

        print("Created forecast", flush=True)
        forecast_rmse = rmse(series, forecast)
        if isinstance(forecast_rmse, list):
            raise ValueError("wtf")
        fixed_rmse = np.float32(forecast_rmse)
        print("Calculated RMSE", flush=True)
        unscaled_forecast = unscaling_pipeline(forecast, scaler, horizon)
        print(f"Pipeline output: length {len(unscaled_forecast)} for period {unscaled_forecast.time_index[0]} to {unscaled_forecast.time_index[-1]}", flush=True)
        finished.set(finished.get()+1)
        
        return Forecast(model.service_id, datetime.now(), model.id, unscaled_forecast, False, fixed_rmse)
    except TimeoutError:
        print(f"Model {model.name} timed out, continuing to next model:", flush=True)
    except Exception as e:
        traceback.print_exc()
        print(f"Model {model.name} failed, continuing to next model: {e}", flush=True)

def validate_model(model: Model) -> bool:
    return model.trained_at > datetime.now() - timedelta(hours=2)

def predict_all(models: list[Model], series: TimeSeries, scaler: Scaler, period: Timedelta, horizon: Timedelta, finished: ValueProxy, cpu_count: int) -> list[Forecast]:
    (n_models, t_models) = split_models(models)
    with Pool(cpu_count) as pool:
        predictions = pool.starmap(predict, [(model, series, scaler, period, horizon, finished) for model in n_models])
    for model in t_models:
        predictions.append(predict(model, series, scaler, period, horizon, finished))
    successful_predictions = []
    for prediction in predictions:
        if prediction is not None:
            successful_predictions.append(prediction)

    return successful_predictions
