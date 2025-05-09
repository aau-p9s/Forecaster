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
import cloudpickle as pickle
from ML.Darts.Utils.preprocessing import run_transformer_pipeline
from datetime import datetime
import pandas as pd

@pytest.fixture
def mock_db():
    """Creates a mock database connection."""
    return MagicMock()

@pytest.fixture
def forecast_repository(mock_db):
    """Creates a ForecastRepository instance with a mocked DB connection."""
    repo = ForecastRepository(mock_db)
    repo.insert_forecast = MagicMock(return_value=None)
    return repo

@pytest.fixture
def model_repository(mock_db, sample_time_series):
    """Creates a ModelRepository instance with a mocked DB connection."""
    model_obj = NaiveSeasonal()
    model = Model("model-id", model_obj, "service")
    model_obj.fit(sample_time_series)
    pickled_model = pickle.dumps(model_obj)
    
    mock_db.get_by_modelid_and_service.return_value = Model("model-id", model_obj, "service")
    mock_db.execute_get.return_value = [("model-id", "model-name", pickled_model)]


    return ModelRepository(mock_db)


@pytest.fixture
def sample_time_series():
    """Creates a datetime-indexed TimeSeries for testing."""
    values = np.random.rand(100)
    start = datetime(2000, 1, 1)
    time_index = pd.date_range(start=start, periods=100, freq='h')  # daily frequency
    return TimeSeries.from_times_and_values(time_index, values)

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

def test_forecaster(forecast_repository, model_repository, sample_time_series):
    data = sample_time_series
    data_processed, missing_values_ratio, scaler = run_transformer_pipeline(data)
    model_obj = NaiveSeasonal()
    model_obj.fit(data_processed[-10:])
    model = Model("model-id", model_obj, "service", scaler)
    models = [model]
    
    forecaster = Forecaster(models, model.serviceId, forecast_repository, model_repository)
    
    forecast = forecaster.create_forecasts(1, data_processed)

    forecast_repository.insert_forecast.assert_called_once()
    
    assert forecast is not None
    # assert isinstance(forecast.forecast, TimeSeries)
    # assert forecast.forecast.n_timesteps == 13
    # assert isinstance(forecast, Forecast)

    # dump = forecast.forecast.to_json()
    # with open("Tests\\forecast.json", "w") as file:
    #     file.write(dump)
