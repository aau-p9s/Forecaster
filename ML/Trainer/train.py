from datetime import datetime
from multiprocessing import Pool
from multiprocessing.managers import DictProxy
from time import time
import traceback

from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.timeseries import TimeSeries
from Database.Entities.Model import Model
from ML.Darts.Utils.split_models import split_models
from ML.Darts.Utils.timeout import timeout


def train_model(model: Model, series: TimeSeries, model_status: DictProxy) -> Model | None:
    try:
        model_status[model.name]["message"] = "working"
        print(f"Training {model.name}", flush=True)
        model_status[model.name]["start_time"] = time()
        if isinstance(model.model, TorchForecastingModel):
            fitted_model = timeout(model.model.fit, series, dataloader_kwargs={ "num_workers": 12 })
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

def train_models(models: list[Model], series: TimeSeries, model_status: DictProxy, cpu_count) -> list[Model]:
    (n_models, t_models) = split_models(models)
    with Pool(cpu_count) as pool:
        trained_models = pool.starmap(train_model, [(model, series.copy(), model_status) for model in n_models])
    for model in t_models:
        trained_models.append(train_model(model, series, model_status))
    successful_models = []
    for model in trained_models:
        if model is not None:
            successful_models.append(model)
    return successful_models

