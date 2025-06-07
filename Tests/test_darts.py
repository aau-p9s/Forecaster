from json import loads
from random import randint
from time import time

from Database.Entities.Forecast import Forecast
from Database.Entities.Historical import Historical
from Database.Entities.Model import Model
from Database.Entities.Service import Service
from Database.Entities.Settings import Settings
from Database.ServiceRepository import ServiceRepository
from Database.Utils import gen_uuid
from ML.Forecaster.Forecaster import Forecaster
from ML.Trainer.Trainer import Trainer
import pytest
import darts.models as models
from darts.models import RegressionEnsembleModel, NaiveEnsembleModel, NaiveSeasonal
from darts.timeseries import TimeSeries
import numpy as np
from ML.Darts.Training.ensemble_training import EnsembleTrainer
from unittest.mock import MagicMock
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from ML.Darts.Utils.preprocessing import run_transformer_pipeline
from datetime import datetime, timedelta
import pandas as pd

settings_id = gen_uuid()
service_id = gen_uuid()
service = Service("My Service", True)
settings = Settings(service.id, 5, 5, 5, 5, pd.to_timedelta("1h"), pd.to_timedelta("1h"))
model = Model("NaiveSeasonal", service.id, NaiveSeasonal(), datetime.now())
now = int(time())
historical = Historical(service.id, datetime.now(), {
    'data':{
        "result":[{
            "values":[
                [now + (i*60), float(randint(0, i))]
                for i in range(60)
            ]
        }]
    }
})

@pytest.fixture
def mock_db():
    """Creates a mock database connection."""
    return MagicMock()

@pytest.fixture
def forecast_repository(mock_db):
    """Creates a ForecastRepository instance with a mocked DB connection."""
    mock_db.insert = MagicMock(return_value=None)
    repo = ForecastRepository(mock_db)
    return repo

@pytest.fixture
def model_repository(sample_time_series:TimeSeries):
    """Creates a ModelRepository instance with a mocked DB connection."""
    model.model.fit(sample_time_series)
    
    model_repository = MagicMock()
    model_repository.get_all_models_by_service.return_value = [ model ]

    return model_repository

@pytest.fixture
def settings_repository():
    repo = MagicMock()
    repo.get_settings.return_value = settings
    return repo

@pytest.fixture
def service_repository():
    service_repository = MagicMock()
    service_repository.all.return_value = [service]
    return service_repository


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

def test_forecaster(forecast_repository:ForecastRepository, model_repository:ModelRepository, settings_repository, sample_time_series:TimeSeries, service_repository:ServiceRepository):
    trainer = Trainer(model.service_id, model_repository, forecast_repository, settings_repository, service_repository)
    trainer._run(historical, pd.to_timedelta("60s"))

    forecaster = Forecaster(model.service_id, model_repository, forecast_repository, settings_repository, service_repository)
    forecast = forecaster._run(historical, pd.to_timedelta("60s"))

    print(forecast.error)

    assert forecast is not None
    assert isinstance(forecast.forecast, TimeSeries)
    assert forecast.forecast.n_timesteps == 61
    assert isinstance(forecast, Forecast)

    # dump = forecast.forecast.to_json()
    # with open("Tests\\forecast.json", "w") as file:
    #     file.write(dump)
