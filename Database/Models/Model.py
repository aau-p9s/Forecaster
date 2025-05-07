import datetime
import tempfile

from darts.models.forecasting.forecasting_model import ForecastingModel

class Model:
    def __init__(self, modelId:str, model:ForecastingModel, serviceId):
        self.modelId = modelId
        self.model = model
        self.trainedTime = datetime.date.today()
        self.serviceId = serviceId
        self.scaler = None

    def get_binary(self):
        temporary_dir = tempfile.mkdtemp()
        
        self.model.save(f"{temporary_dir}/model.pth")
        
        with open(f"{temporary_dir}/model.pth", "rb") as file:
            return file.read()
        
    def get_scaler_binary(self):
        temporary_dir = tempfile.mkdtemp()
        
        self.scaler.save(f"{temporary_dir}/scaler.pth")
        
        with open(f"{temporary_dir}/scaler.pth", "rb") as file:
            return file.read()