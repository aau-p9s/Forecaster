

from multiprocessing import Process
from uuid import UUID

from darts import TimeSeries

from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.Models.Historical import Historical
from Database.Models.Model import Model
from ML.Darts.Utils.preprocessing import load_historical_data, load_json_data, run_transformer_pipeline
from ML.Forecaster import Forecaster


class Trainer:
    def __init__(self, service_id:UUID, model_repository:ModelRepository, forecast_repository:ForecastRepository) -> None:
        self.id:UUID = service_id
        self.model_repository:ModelRepository = model_repository
        self.forecast_repository:ForecastRepository = forecast_repository
        self.forecaster = Forecaster(service_id, model_repository, forecast_repository)

    def train(self, series:list[Historical], horizon:int) -> None:
        self._process:Process = Process(target=self._train, args=[series, horizon])
        self._process.start()

    def _train(self, data:list[Historical], horizon:int) -> None:
        for selected_data in data:
            series:TimeSeries = load_historical_data(selected_data)
            preprocessed_series, missing_value_ratio, scaler = run_transformer_pipeline(series)
            train_series, validation_series = preprocessed_series.split_after(.75)
            print(f"preprocessed_series length: {len(preprocessed_series)}")
            print(f"mssing_value_ratio:         {missing_value_ratio}")
            print(f"scaler:                     {scaler}")

            models = self.model_repository.get_all_models_by_service(self.id)
            for model in models:
                try:
                    fitted_model = model.model.fit(train_series)
                    print("Fitted model")
                    self.model_repository.upsert_model(Model(model.modelId, model.name, fitted_model, model.serviceId, model.scaler))
                    print("Saved model")
                except Exception as e:
                    raise e


        self.forecaster.predict(data, horizon)
