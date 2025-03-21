from Database.ForecastRepository import ForecastRepository
from ML.Model import Model
from darts import TimeSeries
from darts.metrics import rmse
from darts.models.forecasting.forecasting_model import ForecastingModel

class Forecast:
    def __init__(self, modelId, forecast: TimeSeries, error=float('inf')):
        self.modelId = modelId
        self.forecast = forecast
        self.error = error

class Forecaster: # Each service has one of these to create / keep track of forecasts
    forecasts = []
    
    def __init__(self, models: list[Model], serviceId, repository:ForecastRepository):
        self.models = models
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
            forecast = model.binary.predict(forecastHorizon)
            forecast_error = rmse(historicalData, forecast)
            self.forecasts.append(Forecast(model.modelId, forecast, forecast_error))

        forecast = self.find_best_forecast()
        print(f"{forecast.modelId=}")
        print(f"{forecast.error}")
        #self.repository.insert_forecast(forecast.modelId, forecast.forecast, forecast.error)
        return forecast

    def find_best_forecast(self): # forecast ranker
        """Finds the forecast with the lowest error and assumes that it is the best"""
        return min(self.forecasts, key=lambda x: x.error)
