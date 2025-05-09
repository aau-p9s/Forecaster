import pytest
import darts.models as models
from darts.models import RegressionEnsembleModel, NaiveEnsembleModel, NaiveSeasonal
from darts.timeseries import TimeSeries
import numpy as np
from ML.Darts.Training.ensemble_training import EnsembleTrainer
from darts.datasets import AirPassengersDataset
from ML.Forecaster import Forecaster, Forecast
from unittest.mock import MagicMock
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.Models.Model import Model
import pickle

@pytest.fixture
def mock_db():
    """Creates a mock database connection."""
    return MagicMock()

@pytest.fixture
def forecast_repository(mock_db):
    """Creates a ForecastRepository instance with a mocked DB connection."""
    mock_db.insert_forecast.return_value = None
    return ForecastRepository(mock_db)

@pytest.fixture
def model_repository(mock_db):
    """Creates a ModelRepository instance with a mocked DB connection."""
    mock_db.execute_get.return_value = []
    return ModelRepository(mock_db)


@pytest.fixture
def sample_time_series():
    """Creates a sample TimeSeries for testing."""
    values = np.random.rand(100)
    return TimeSeries.from_values(values)

@pytest.fixture
def pre_trained_local_models(sample_time_series):
    """Creates and trains two sample forecasting models."""
    model1 = models.ExponentialSmoothing()
    model2 = models.ARIMA()
    
    model1.fit(sample_time_series)
    model2.fit(sample_time_series)
    
    return [model1, model2]

@pytest.fixture
def pre_trained_global_models(sample_time_series):
    """Creates and trains two sample forecasting models."""
    model3 = models.NHiTSModel(12,12, n_epochs=20)
    model4 = models.TSMixerModel(12,12, n_epochs=20)
    
    model3.fit(sample_time_series)
    model4.fit(sample_time_series)
    
    return [model3, model4]


@pytest.fixture
def ensemble_training_local(pre_trained_local_models, sample_time_series):
    """Returns an instance of EnsembleTraining with pre-trained models."""
    train_series, val_series = sample_time_series.split_after(0.75)
    return EnsembleTrainer(pre_trained_local_models, train_series, val_series, forecast_period=12)

@pytest.fixture
def ensemble_training_global(pre_trained_global_models, sample_time_series):
    """Returns an instance of EnsembleTraining with pre-trained models."""
    train_series, val_series = sample_time_series.split_after(0.75)
    return EnsembleTrainer(pre_trained_global_models, train_series, val_series, forecast_period=12)


def test_learned_ensemble_model(ensemble_training_global):
    """Test RegressionEnsembleModel functionality."""
    rmse_error, backtest, model = ensemble_training_global.create_learned_ensemble_model()
    
    assert isinstance(model, RegressionEnsembleModel)
    assert backtest is not None
    assert isinstance(rmse_error, float) and rmse_error is not None

def test_naive_ensemble_model(ensemble_training_local):
    """Test NaiveEnsembleModel functionality."""
    rmse_error, backtest, model = ensemble_training_local.create_naive_ensemble_model()
    
    assert all(model._fit_called for model in ensemble_training_local.candidate_models)

    assert isinstance(model, NaiveEnsembleModel)
    assert backtest is not None
    assert isinstance(rmse_error, float) and rmse_error >= 0

def test_forecaster(mock_db, forecast_repository, model_repository):
    data = AirPassengersDataset().load()

    model_obj = NaiveSeasonal()
    model_obj.fit(data[-10:])
    model = Model("model-id", model_obj, "service")
    models = [model]
    
    forecaster = Forecaster(models, model.serviceId, forecast_repository, model_repository)
    
    forecast = forecaster.create_forecasts(13, data)

    mock_db.insert_forecast.assert_called_once_with(forecast, service_id)
    assert forecast is not None
    assert isinstance(forecast.forecast, TimeSeries)
    assert forecast.forecast.n_timesteps == 13
    assert isinstance(forecast, Forecast)

    # dump = forecast.forecast.to_json()
    # with open("Tests\\forecast.json", "w") as file:
    #     file.write(dump)
