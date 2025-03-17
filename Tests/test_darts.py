import re
import pytest
import darts.models as models
from darts.models import RegressionEnsembleModel, NaiveEnsembleModel
from darts.metrics import rmse
from darts.timeseries import TimeSeries
import numpy as np
from ML.Darts.Training.ensemble_training import EnsembleTrainer

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
    return EnsembleTraining(pre_trained_local_models, sample_time_series, forecast_period=12)

@pytest.fixture
def ensemble_training_global(pre_trained_global_models, sample_time_series):
    """Returns an instance of EnsembleTraining with pre-trained models."""
    return EnsembleTraining(pre_trained_global_models, sample_time_series, forecast_period=12)


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