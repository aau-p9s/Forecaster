import pytest
from unittest.mock import MagicMock
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
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
    assert result[0][1] == "MyModel"  # Ensure the model name is correct

def test_insert_model(mock_db):
    """Test inserting a model with a mocked database"""
    mock_db.execute_query.return_value = [
        ("1234", "NewModel", b"binary_data", "2024-02-05 12:30:00", "service_2")
    ]

    model_repo = ModelRepository(mock_db)
    result = model_repo.insert_model("NewModel", "Tests/test_db.py", "2024-02-05 12:30:00", "service_2")

    assert result[0][1] == "NewModel"  # Check inserted model name

def test_get_latest_forecast(mock_db):
    """Test getting the latest forecast"""
    mock_db.execute_query.return_value = [
        ("5678", "1234", 0.95, "service_1")
    ]

    forecast_repo = ForecastRepository(mock_db)
    result = forecast_repo.get_latest_forecast("1234", "service_1")

    mock_db.execute_query.assert_called_once_with(
        'SELECT * FROM forecasts WHERE "modelid" = %s AND "serviceid" = %s ORDER BY "createdat" DESC LIMIT 1;',
        ("1234", "service_1")
    )
    assert result[0] == ("5678", "1234", 0.95, "service_1")  # Ensure forecast ID matches
