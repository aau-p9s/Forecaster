
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from Database.Entities.Model import Model


def split_models(models: list[Model]) -> tuple[list[Model], list[Model]]:
    normal_models = []
    torch_models = []
    for model in models:
        if isinstance(model.model, TorchForecastingModel):
            torch_models.append(model)
        elif isinstance(model.model, ForecastingModel):
            normal_models.append(model)
        else:
            raise ValueError(f"Huh? {model.name} isn't a forecasting model")
    return (normal_models, torch_models)
