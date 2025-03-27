import csv
import pickle
from numpy import NaN, nan
from darts import TimeSeries
import numpy
from Database.ForecastRepository import ForecastRepository
from darts.metrics import rmse
from Database.Models.Forecast import Forecast
from Database.Models.Model import Model

class Forecaster: # Each service has one of these to create / keep track of forecasts
    forecasts:list[Forecast] = []
    
    def __init__(self, models: list[Model], serviceId, repository:ForecastRepository):
        self.models:list[Model] = models
        self.serviceId = serviceId
        self.repository = repository
    
    def create_forecasts(self, forecastHorizon, historicalData=None) -> Forecast:
        """Creates a forecast for with each supplied model and calculates its error by backtesting
        Args:
          historicalData (TimeSeries): Used to backtest and supply timestamp where to predict from
        Returns:
            str: Best forecast.
        """
        for model in self.models:
            # Use predict from Darts and backtest to calculate errors for models on historical data here
            forecast = model.model.predict(forecastHorizon)
            # TODO: use real data
            if historicalData is None:
                historicalData = TimeSeries.from_csv("./Assets/test_data.csv")
            forecast_error = rmse(historicalData, forecast)
            self.forecasts.append(Forecast(model.modelId, forecast, forecast_error))

        forecast = self.find_best_forecast()
        print(f"{forecast.modelId=}")
        print(f"{forecast.error=}")
        self.repository.insert_forecast(forecast, self.serviceId)
        return forecast

    def find_best_forecast(self): # forecast ranker
        """Finds the forecast with the lowest error and assumes that it is the best"""
        return min(self.forecasts, key=lambda x: x.error)
