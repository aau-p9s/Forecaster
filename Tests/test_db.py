from pickle import loads
from darts.timeseries import TimeSeries
import pytest
from unittest.mock import MagicMock
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.Models.Forecast import Forecast
from Database.Models.Model import Model

@pytest.fixture
def mock_db():
    """Fixture to create a mock database connection"""
    mock_db = MagicMock()
    mock_db.execute_get = MagicMock()
    return mock_db

def test_get_all_models_by_service(mock_db):
    """Test fetching models for a service with a mocked database"""
    with open("Assets/test_model.pth", "rb") as file:
        modelObj = loads(file.read())
    model = Model("model", modelObj, "service")
    mock_db.execute_query.return_value = [
        ("model-id", type(modelObj).__name__, model.get_binary(), model.serviceId)
    ]

    model_repo = ModelRepository(mock_db)
    result = model_repo.get_all_models_by_service("service")

    mock_db.execute_query.assert_called_once_with(
        'SELECT id, name, bin from models WHERE "serviceid" = %s;', [model.serviceId]
    )
    assert result[0] == model # Ensure the model is equal in the database

def test_insert_model(mock_db):
    """Test inserting a model with a mocked database"""
    mock_db.execute_query.return_value = [
        ("name")
    ]

    model_repo = ModelRepository(mock_db)
    with open("Assets/test_model.pth", "rb") as file:
        model = loads(file.read())
    result = model_repo.insert_model(Model("NewModel", model, "myservice"))

    assert result == model # Check inserted model name

def test_get_latest_forecast(mock_db):
    """Test getting the latest forecast"""
    data = TimeSeries.from_csv("./Assets/test_data.csv")
    forecast = Forecast("forecast-id", data)

    mock_db.execute_query.return_value = [
            ("model-id", forecast.serialize())
    ]

    forecast_repo = ForecastRepository(mock_db)
    result = forecast_repo.get_latest_forecast("testforecast", "service")

    mock_db.execute_query.assert_called_once_with(
        'SELECT modelid, forecast FROM forecasts WHERE "modelid" = %s AND "serviceid" = %s ORDER BY "createdat" DESC LIMIT 1;',
        [ "model-id", "service" ]

    )
    assert result == Forecast("testforecast", data)  # Ensure forecast ID matches
