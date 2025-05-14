from json import dumps, loads
from multiprocessing import Process
from flask import Response
from flask_restx import Resource
from datetime import datetime

from ML.Trainer import Trainer
from ..lib.variables import model_repository, api, trainers, forecast_repository, historical_repository

def format_data(data):
    return {
        "timestamp":[datetime.fromtimestamp(value[0]) for value in data["data"]["result"][0]["values"]],
        "value":[float(value[1]) for value in data["data"]["result"][0]["values"]]
    }

@api.route("/train/<string:serviceId>/<forecastHorizon>")
class Train(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 202:"working...", 500:"Something ML died!!!!"})
    def post(self, serviceId, forecastHorizon=12):
        # Retrain models on new thread and predict + copy to DB
        models = model_repository.get_all_models_by_service(serviceId)
        if not serviceId in trainers:
            historical = historical_repository.get_by_service(serviceId)
            data = format_data(historical[0].data)
            trainData = {"tuning_data": data}
            trainer = Trainer(models, serviceId, trainData, forecastHorizon, model_repository, forecast_repository)
            trainers[serviceId] = {
                "trainer": trainer,
                "thread": Process(target=trainer.train_model)
            }

        thread:Process = trainers[serviceId]["thread"]
        if thread.is_alive():
            thread.join()
            return Response(status=202, response=dumps({"message":f"Training was already in progress for {serviceId}"}))

        trainer:Trainer = trainers[serviceId]["trainer"]
        thread = Process(target=trainer.train_model)
        trainers[serviceId]["thread"] = thread
        thread.start()
        thread.join()

        return Response(status=200, response=dumps({"message":f"Training started for {serviceId}"}))


