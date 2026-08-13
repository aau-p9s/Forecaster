from datetime import datetime
from darts.models import NaiveSeasonal
import pytest
from unittest.mock import MagicMock
from Database.Entities.Model import Model
from Database.Entities.Historical import Historical
from Database.Entities.Service import Service
from Database.ModelRepository import ModelRepository
from Database.HistoricDataRepository import HistoricDataRepository

service = Service("My Service", True)
model = Model("NaiveSeasonal", service.id, NaiveSeasonal(), datetime.now())
historical = Historical(service.id, datetime.now(), {"data":"xd"})

@pytest.fixture
def mock_db():
    """Fixture to create a mock database connection"""
    mock_db = MagicMock()
    mock_db.execute_get = MagicMock()
    return mock_db

#def test_get_all_models_by_service(mock_db):
#    """Test fetching models for a service with a mocked database"""
#    mock_db.execute_get.return_value = [
#        
#    ]
#
#    model_repo = ModelRepository(mock_db)
#    result = model_repo.get_all_models_by_service(service.id)
#
#    mock_db.execute_get.assert_called_once_with(
#        'SELECT id, name, bin, ckpt from models WHERE serviceid = %s;', [str(model.service_id)]
#    )
#    assert result[0].get_binary() == model.get_binary() # Ensure the model is equal in the database

# def test_insert_model(mock_db):
#     """Test inserting a model with a mocked database"""

#     data = AirPassengersDataset().load()
#     model_obj = NaiveSeasonal()
#     model_obj.fit(data[-10:])
#     model = Model("model-id", model_obj, "service")

#     mock_db.execute_get.return_value = [
#         (model.modelId, type(model_obj).__name__, model.get_binary(), model.serviceId)
#     ]

#     model_repo = ModelRepository(mock_db)
#     result = model_repo.insert_model(model)

#     assert result.get_binary() == model.get_binary() # Check inserted model

def test_get_historical(mock_db):
    """Test getting the latest forecast"""
    mock_db.execute_get.return_value = [
        historical.to_row()
    ]
    
    repo = HistoricDataRepository(mock_db)
    result = repo.get_by_service(service.id)

    assert result[0].data == historical.data
