from pickle import loads
from darts.datasets import AirPassengersDataset
from darts.models import NaiveSeasonal
import pytest
from unittest.mock import MagicMock
from Database.ForecastRepository import ForecastRepository
from Database.HistoricalRepository import HistoricalRepository
from Database.ModelRepository import ModelRepository
from Database.Models.Forecast import Forecast
from Database.Models.Historical import Historical
from Database.Models.Model import Model

@pytest.fixture
def mock_db():
    """Fixture to create a mock database connection"""
    mock_db = MagicMock()
    mock_db.execute_get = MagicMock()
    return mock_db

def test_get_all_models_by_service(mock_db):
    """Test fetching models for a service with a mocked database"""
    model_obj = NaiveSeasonal()
    model = Model("model-id", model_obj, "service")
    mock_db.execute_get.return_value = [
        ("model-id", type(model_obj).__name__, model.get_binary(), "minmax", model.serviceId)
    ]
    mock_db.execute_get.return_value = [
        ("model-id", type(model_obj).__name__, model.get_binary(), model.serviceId)
    ]

    model_repo = ModelRepository(mock_db)
    result = model_repo.get_all_models_by_service("service")

    mock_db.execute_get.assert_called_once_with(
        'SELECT id, name, bin from models WHERE "serviceid" = %s;', [model.serviceId]
    )
    assert result[0].get_binary() == model.get_binary() # Ensure the model is equal in the database

# def test_insert_model(mock_db):
#     """Test inserting a model with a mocked database"""
#     with open("Assets/test_model.pth", "rb") as file:
#         modelObj = loads(file.read())
#     model = Model("model-id", modelObj, "service")

#     mock_db.execute_get.return_value = [
#         (model.modelId, type(modelObj).__name__, model.get_binary(), model.serviceId)
#     ]

#     model_repo = ModelRepository(mock_db)
#     result = model_repo.insert_model(model)

#     assert result.get_binary() == model.get_binary() # Check inserted model

def test_get_historical(mock_db):
    """Test getting the latest forecast"""
    historical = Historical("historical-id", "service", 0.0, "{'data':'xd'}")
    mock_db.execute_get.return_value = [
        (historical.id, historical.service_id, historical.timestamp, historical.data)
    ]
    
    repo = HistoricalRepository(mock_db)
    result = repo.get_by_service("service")

    assert result[0].data == historical.data
