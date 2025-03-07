import darts.models as models
from darts.models import RegressionEnsembleModel, NaiveEnsembleModel
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.forecasting_model import LocalForecastingModel
from darts.metrics import rmse
import inspect

# An ensemble model should be created using the best models found in the tuning process to check if the ensemble model can improve the forecast
class EnsembleTraining:

    def __init__(self, candidate_models, series, forecast_period):
        self.candidate_models = candidate_models
        self.series = series
        self.forecast_period = forecast_period
        
    def learned_ensemble_model(self, regression_train_n_points=12):
        """Regression ensemble model that uses forecasts from the candidate models as features to train a regression model to create a better forecast. WORKS ONLY FOR PRE-TRAINED (fitted) GlobalForecastingModels"""
        
        for model in self.candidate_models:
            if model._model is None:
                raise ValueError(f"Model {model} is not pre-trained. Train it first.")

        ensemble_model = RegressionEnsembleModel(
            forecasting_models=self.candidate_models,
            regression_train_n_points=regression_train_n_points,
        )

        ensemble_model.fit(self.series)

        backtest = ensemble_model.historical_forecasts(self.series, forecast_horizon=self.forecast_period)

        rmse_error = rmse(backtest, self.series)

        return (rmse_error, backtest, ensemble_model) # Should save the backtest (forecast) to the database along with the model and the error

    def naive_ensemble_model(self):
        """"Naive ensemble model that averages the forecasts of the candidate models to create a better forecast."""
        
        for model in self.candidate_models:
            if model._model is None:
                raise ValueError(f"Model {model} is not pre-trained. Train it first.")
        
        ensemble_model = NaiveEnsembleModel(
            forecasting_models=self.candidate_models
        )

        backtest = ensemble_model.historical_forecasts(self.series, forecast_horizon=self.forecast_period)
        rmse_error = rmse(backtest, self.series)

        return (rmse_error, backtest, ensemble_model) # Should save the backtest (forecast) to the database along with the model and the error