import pytest
import darts.models as models
from darts.models import RegressionEnsembleModel, NaiveEnsembleModel
from darts.metrics import rmse
from darts.timeseries import TimeSeries
import numpy as np
from ML.Darts.Training.EnsembleTraining import EnsembleTraining   # Assuming your class is saved in ensemble_training.py

@pytest.fixture
def sample_time_series():
    """Creates a sample TimeSeries for testing."""
    values = np.random.rand(100)
    return TimeSeries.from_values(values)

@pytest.fixture
def pre_trained_models(sample_time_series):
    """Creates and trains two sample forecasting models."""
    model1 = models.ExponentialSmoothing()
    model2 = models.ARIMA()
    
    model1.fit(sample_time_series)
    model2.fit(sample_time_series)
    
    return [model1, model2]

@pytest.fixture
def ensemble_training(pre_trained_models, sample_time_series):
    """Returns an instance of EnsembleTraining with pre-trained models."""
    return EnsembleTraining(pre_trained_models, sample_time_series, forecast_period=12)


def test_learned_ensemble_model(ensemble_training):
    """Test RegressionEnsembleModel functionality."""
    rmse_error, backtest, model = ensemble_training.learned_ensemble_model()
    
    assert isinstance(model, RegressionEnsembleModel)
    assert backtest is not None
    assert isinstance(rmse_error, float) and rmse_error >= 0

def test_naive_ensemble_model(ensemble_training):
    """Test NaiveEnsembleModel functionality."""
    rmse_error, backtest, model = ensemble_training.naive_ensemble_model()
    
    assert isinstance(model, NaiveEnsembleModel)
    assert backtest is not None
    assert isinstance(rmse_error, float) and rmse_error >= 0

def test_untrained_model_error(sample_time_series):
    """Ensure an error is raised if models are not trained before being passed to the ensemble."""
    model1 = models.ExponentialSmoothing()
    model2 = models.ARIMA()
    
    ensemble = EnsembleTraining([model1, model2], sample_time_series, forecast_period=12)
    
    with pytest.raises(ValueError, match="is not pre-trained. Train it first."):
        ensemble.learned_ensemble_model()
    
    with pytest.raises(ValueError, match="is not pre-trained. Train it first."):
        ensemble.naive_ensemble_model()
