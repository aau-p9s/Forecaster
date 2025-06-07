from datetime import datetime
from multiprocessing import Pool, cpu_count
from multiprocessing.managers import DictProxy
from time import time
from math import ceil
import traceback

from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.timeseries import TimeSeries
from Database.Entities.Model import Model
from ML.Darts.Utils.timeout import timeout
from Utils.variables import service_repository


def train_model(model: Model, series: TimeSeries, model_status: DictProxy) -> Model | None:
    try:
        model_status[model.name]["message"] = "working"
        print(f"Training {model.name}", flush=True)
        model_status[model.name]["start_time"] = time()
        if isinstance(model.model, TorchForecastingModel):
            fitted_model = timeout(model.model.fit, dataloader_kwargs={ "num_workers": 12 })
        else:
            fitted_model = timeout(model.model.fit, series)
        model_status[model.name]["message"] = "finished"
        model_status[model.name]["end_time"] = time()
        print(f"Finished training {model.name}")
        return Model(model.name, model.service_id, fitted_model, datetime.now(), model.scaler).with_id(model.id)

    except Exception as e:
        model_status[model.name]["end_time"] = time()
        model_status[model.name]["message"] = "failed"
        model_status[model.name]["error"] = f"{e}"
        traceback.print_exc()
        return None

def train_models(models: list[Model], series: TimeSeries, model_status: DictProxy) -> list[Model]:
    service_count = len(list(filter(lambda service: service.autoscaling_enabled, service_repository.all())))
    with Pool(ceil(cpu_count()/(service_count*2))) as pool:
        trained_models = pool.starmap(train_model, [(model, series.copy(), model_status) for model in models])
    successful_models = []
    for model in trained_models:
        if model is not None:
            successful_models.append(model)
    return successful_models

