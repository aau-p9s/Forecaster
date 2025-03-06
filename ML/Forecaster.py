class Forecast:
    def __init__(self, modelId, forecast, error=float('inf')):
        self.modelId = modelId
        self.forecast = forecast
        self.error = error

class Forecaster: # Each service has one of these to create / keep track of forecasts
    forecasts = {Forecast("modelId", '''
                2025-02-05 12:00:00,120
                2025-02-05 12:01:00,135
                2025-02-05 12:02:00,128
                2025-02-05 12:03:00,140
                2025-02-05 12:04:00,150
                2025-02-05 12:05:00,145
                2025-02-05 12:06:00,160
                2025-02-05 12:07:00,155
                ''', 0.105424)} # This is just a sample
    
    def __init__(self, models, serviceId):
        self.models = models
        self.serviceId = serviceId
    
    def create_forecasts(self, historicalData):
        """Creates a forecast for with each supplied model and calculates its error by backtesting
        Args:
          historicalData (TimeSeries): Used to backtest and supply timestamp where to predict from
        Returns:
            str: Best forecast.
        """
        self.forecasts.clear()
        for model in self.models:
            # Use predict from Darts and backtest to calculate errors for models on historical data here
            forecast = None # This is returned from Darts predict function
            forecast_error = None # This is returned from backtesting function
            self.forecasts.add(Forecast(model.id, forecast, forecast_error))
        return self.find_best_forecast()

    def find_best_forecast(self): # forecast ranker
        """Finds the forecast with the lowest error and assumes that it is the best"""
        return min(self.forecasts, key=lambda x: x[1])
