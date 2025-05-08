from darts import TimeSeries
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from darts.metrics import rmse
from Database.Models.Forecast import Forecast
from Database.Models.Model import Model

class Forecaster: # Each service has one of these to create / keep track of forecasts
    forecasts:list[Forecast] = []
    
    def __init__(self, models: list[Model], serviceId, repository:ForecastRepository, model_repository:ModelRepository):
        self.models:list[Model] = models
        self.serviceId = serviceId
        self.repository = repository
        self.model_repository = model_repository
    
    def create_forecasts(self, forecastHorizon, historicalData=None):
        """Creates a forecast for each supplied model and calculates its error by backtesting
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
            forecast_error = rmse(historicalData, forecast, intersect=True)
            forecast = Forecast(model.modelId, forecast, forecast_error)
            self.forecasts.append(forecast)

            self.repository.insert_forecast(forecast, self.serviceId) #Maybe shouldn't insert all forecasts, but only the best one
            print("Forecast inserted in db")
        best_forecast = self.find_best_forecast()
        print(f"Best forecast: {best_forecast.modelId} with error {best_forecast.error}")
        best_forecast_model = self.model_repository.get_by_modelid_and_service(best_forecast.modelId, self.serviceId)
        best_forecast = Forecast(best_forecast_model.modelId, best_forecast.inverse_scale(best_forecast_model.scaler), best_forecast.error)
        return best_forecast

    def find_best_forecast(self): # forecast ranker
        """Finds the forecast with the lowest error and assumes that it is the best"""
        return min(self.forecasts, key=lambda x: x.error)
