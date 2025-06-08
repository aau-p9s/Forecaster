from pandas import Timedelta
from Database.Entities.Historical import Historical
from ML.Darts.Utils.MLManager import MLManager
from ML.Forecaster.Forecaster import Forecaster
from ML.Trainer.train import train_models

class Trainer(MLManager):
    def __init__(self, *args, **kwargs):
        self.forecaster = Forecaster(*args, **kwargs)
        super().__init__(*args, **kwargs)
        self.model_status = self.manager.dict()

    def _run(self, historical:Historical, horizon:Timedelta, gpu_id: int = 0) -> None:
        self.model_status.clear()
        self.busy("Training")
        train_series, _, _ = self.preprocess(historical, horizon)

        print("Getting models", flush=True)
        models = self.model_repository.get_all_models_by_service(self.service_id, gpu_id=gpu_id)
        print("Got models", flush=True)
        for model in models:
            self.model_status[model.name] = self.manager.dict({ "message": "waiting", "error": None, "start_time": None, "end_time": None })

        fitted_models = train_models(models, train_series, self.model_status, self.get_cores())

        print("Finished training", flush=True)

        for fitted_model in fitted_models:
            if fitted_model is None:
                continue
            self.model_status[fitted_model.name]["message"] = "saving"
            print("Saving model...", flush=True)
            self.model_repository.upsert_model(fitted_model)
            self.model_status[fitted_model.name]["message"] = "saved"
        
        self.idle()

