from pickle import UnpicklingError
import tempfile
import traceback

from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
import torch

from Utils.variables import temporary_directory, enable_gpu



def load_model(name: str, data: bytes, ckpt: bytes|None = None, gpu_id: int = 0) -> ForecastingModel:
    with tempfile.TemporaryDirectory(dir=temporary_directory) as directory:
        if ckpt is not None:
            with open(f"{directory}/{name}.pth.ckpt", "wb") as file:
                file.write(ckpt)
        with open(f"{directory}/{name}.pth", "wb") as file:
            file.write(data)
        try:
            device = torch.device(f"cuda:{gpu_id}" if enable_gpu else "cpu")
            model = TorchForecastingModel.load(f"{directory}/{name}.pth", map_location=device)
            if not enable_gpu:
                model.to_cpu()
            else:
                model.trainer_params['devices'] = [gpu_id]
                model.trainer_params['accelerator'] = "gpu"
        except Exception as e1:
            try:
                model = ForecastingModel.load(f"{directory}/{name}.pth")
            except Exception as e2:
                traceback.print_exception(e1)
                traceback.print_exception(e2)
                raise UnpicklingError
        return model 

