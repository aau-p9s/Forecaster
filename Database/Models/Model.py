import datetime
import tempfile
from ML.Darts.Utils.preprocessing import ScalerType

from darts.models.forecasting.forecasting_model import ForecastingModel

class Model:
    def __init__(self, modelId:str, model:ForecastingModel, serviceId, scaler=ScalerType.MINMAX):
        self.modelId = modelId
        self.model = model
        self.trainedTime = datetime.date.today()
        self.serviceId = serviceId
        self.scaler = scaler

    def get_binary(self):
        temporary_dir = tempfile.mkdtemp()
        
        self.model.save(f"{temporary_dir}/model.pth")
        
        with open(f"{temporary_dir}/model.pth", "rb") as file:
            return file.read()