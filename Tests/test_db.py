from pickle import loads
from darts.datasets import AirPassengersDataset
from darts.timeseries import TimeSeries
import pytest
from unittest.mock import MagicMock
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.Models.Forecast import Forecast
from Database.Models.Model import Model
from Database.SettingsRepository import SettingsRepository
from Database.ServiceRepository import ServiceRepository

@pytest.fixture
def mock_db():
    """Fixture to create a mock database connection"""
    mock_db = MagicMock()
    mock_db.execute_query = MagicMock()
    return mock_db

def test_get_all_models_by_service(mock_db):
    """Test fetching models for a service with a mocked database"""
    mock_db.execute_query.return_value = [
        ("1234", "MyModel", b"binary_data", "2024-02-05 12:00:00", "service_1")
    ]

    model_repo = ModelRepository(mock_db)
    result = model_repo.get_all_models_by_service("service_1")

    mock_db.execute_query.assert_called_once_with(
        'SELECT * from models WHERE "serviceid" = %s;', ("service_1",)
    )
    assert len(result) == 1
    assert type(result[0].model).__name__ == "MyModel"  # Ensure the model name is correct

def test_insert_model(mock_db):
    """Test inserting a model with a mocked database"""
    mock_db.execute_query.return_value = [
        ("1234", "NewModel", b"binary_data", "2024-02-05 12:30:00", "service_2")
    ]

    model_repo = ModelRepository(mock_db)
    with open("Assets/autotheta_model.pth", "rb") as file:
        model = loads(file.read())
    result = model_repo.insert_model(Model("NewModel", model, "myservice"))

    assert result.modelId == "NewModel"  # Check inserted model name

def test_get_latest_forecast(mock_db):
    """Test getting the latest forecast"""
    data = TimeSeries.from_csv("./test_data.csv")

    mock_db.execute_query.return_value = Forecast("testforecast", data)

    forecast_repo = ForecastRepository(mock_db)
    result = forecast_repo.get_latest_forecast("testforecast", "service_1")

    mock_db.execute_query.assert_called_once_with(
        'SELECT * FROM forecasts WHERE "modelid" = %s AND "serviceid" = %s ORDER BY "createdat" DESC LIMIT 1;',
        ("testforecast", "service_1")
    )
    assert result == Forecast("testforecast", data)  # Ensure forecast ID matches
