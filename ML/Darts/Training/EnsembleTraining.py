import darts.models as models
from darts.models import RegressionEnsembleModel, NaiveEnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.metrics import rmse
from darts.timeseries import TimeSeries
import numpy as np
import inspect

# An ensemble model should be created using the best models found in the tuning process to check if the ensemble model can improve the forecast
class EnsembleTraining:

    def __init__(self, candidate_models: list[ForecastingModel], series, forecast_period):
        self.candidate_models = candidate_models
        self.series = series
        self.forecast_period = forecast_period
        
    def create_learned_ensemble_model(self, regression_train_n_points=12):
        """Regression ensemble model that uses forecasts from the candidate models as features to train a regression model to create a better forecast. WORKS ONLY FOR PRE-TRAINED (fitted) GlobalForecastingModels"""
        
        all_models_equally_fit = (all(model._fit_called for model in self.candidate_models) or all(not model._fit_called for model in self.candidate_models))
        
        if not all_models_equally_fit:
            raise ValueError("All models must be either pre-trained or not pre-trained.")
        
        for model in self.candidate_models:
            if not isinstance(model, GlobalForecastingModel):
                raise ValueError(f"Model {model} is not a GlobalForecastingModel. Only GlobalForecastingModels are supported. Use Naive instead")
            
        ensemble_model = RegressionEnsembleModel(
            forecasting_models=self.candidate_models,
            regression_train_n_points=regression_train_n_points,
            train_forecasting_models=False
        )

        ensemble_model.fit(self.series)

        #forecast = ensemble_model.backtest(self.series, forecast_horizon=self.forecast_period)
        forecast = ensemble_model.predict(self.forecast_period)
        rmse_error = rmse(forecast, self.series)

        return (rmse_error, forecast, ensemble_model) # Should save the backtest (forecast) to the database along with the model and the error

    def create_naive_ensemble_model(self):
        """"Naive ensemble model that averages the forecasts of the candidate models to create a better forecast."""
        
        all_models_equally_fit = (all(model._fit_called for model in self.candidate_models) or all(not model._fit_called for model in self.candidate_models))
        
        if not all_models_equally_fit:
            raise ValueError("All models must be either pre-trained or not pre-trained.")

        # Check if all models in candidate_models are instances of LocalForecastingModel
        if all(isinstance(model, LocalForecastingModel) for model in self.candidate_models):
            ensemble_model = NaiveEnsembleModel(
                forecasting_models=self.candidate_models,
                train_forecasting_models=True
            )
        elif all(isinstance(model, GlobalForecastingModel) for model in self.candidate_models) and all_models_equally_fit:
            ensemble_model = NaiveEnsembleModel(
                forecasting_models=self.candidate_models,
                train_forecasting_models=False
            )
        else:
            ensemble_model = NaiveEnsembleModel(
                forecasting_models=self.candidate_models,
                train_forecasting_models=True
            )
        

        #forecast = ensemble_model.backtest(self.series, forecast_horizon=self.forecast_period)
        forecast = ensemble_model.predict(self.forecast_period)
        rmse_error = rmse(forecast, self.series)

        return (rmse_error, forecast, ensemble_model) # Should save the backtest (forecast) to the database along with the model and the error
    
if __name__ == "__main__":
    values = np.random.rand(100)
    series = TimeSeries.from_values(values)

    model1 = models.ExponentialSmoothing()
    model2 = models.ARIMA()
    model3 = models.NHiTSModel(12,12)
    model4 = models.TSMixerModel(12,12)
    model1.fit(series)
    model2.fit(series)
    #model3.fit(series)
    #model4.fit(series)
    #learned_ensemble = EnsembleTraining([model3, model4], series, 12)
    naive_ensemble = EnsembleTraining([model1, model2], series, 12)
    naive_ensemble.create_naive_ensemble_model()
    #print(learned_ensemble.create_learned_ensemble_model())