from darts import TimeSeries
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from darts.metrics import rmse
from Database.Models.Forecast import Forecast
from Database.Models.Historical import Historical
from Database.Models.Model import Model
from sklearn.preprocessing import MinMaxScaler
from ML.Darts.Utils.preprocessing import Scaler

class Forecaster: # Each service has one of these to create / keep track of forecasts
    forecasts:list[Forecast] = []
    
    def __init__(self, models: list[Model], serviceId, repository:ForecastRepository, model_repository:ModelRepository):
        self.models:list[Model] = models
        self.serviceId = serviceId
        self.repository = repository
        self.model_repository = model_repository
    
    def create_forecasts(self, forecastHorizon:int, historicalData:Historical=None):
        """Creates a forecast for each supplied model and calculates its error by backtesting
        Args:
          historicalData (Historical): Used to backtest and supply timestamp where to predict from
        Returns:
            str: Best forecast.
        """
        for model in self.models:
            # Use predict from Darts and backtest to calculate errors for models on historical data here
            try:
                try:
                    print(f"Creating forecast for {model.name}\n")
                    forecast = model.model.predict(forecastHorizon)
                except Exception as e:
                    print(f"Error predicting with model {model.name}: {str(e)}")
                    raise e
                if historicalData is None or len(historicalData.data) == 0:
                    raise ValueError("No historical data provided for backtesting.")
                    
                print(f"Calculating rmse for {len(historicalData)} and {len(forecast)}")
                forecast_error = rmse(historicalData.data, forecast)
                print(forecast_error)
                forecast = Forecast(model.modelId, forecast, forecast_error)
                self.forecasts.append(forecast)

            except Exception as e:
                print(f"Error creating forecast for {model.name}: {str(e)}")
                continue
            best_forecast = self.find_best_forecast()
            print("Forecast inserted in db")

        if not best_forecast:
            raise ValueError("No forecasts available to find the best one.")
        best_forecast.inverse_scale(scaler)
        self.repository.insert_forecast(forecast, self.serviceId) #Saves the forecast to the database
        
        print(f"Best forecast: {best_forecast.modelId} with error {best_forecast.error}")

        scaler = Scaler(MinMaxScaler(feature_range=(0, 1)))
        scaler.fit(historicalData)

    def find_best_forecast(self): # forecast ranker
        """Finds the forecast with the lowest error and assumes that it is the best"""
        if not self.forecasts:
            raise ValueError("No forecasts available to find the best one.")
        return min(self.forecasts, key=lambda x: x.error)
