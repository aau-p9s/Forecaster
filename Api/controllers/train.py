from json import dumps
from uuid import UUID
from flask import Response
from flask_restx import Resource
import pandas as pd

from Database.Entities.Historical import Historical
from ML.Trainer.Trainer import Trainer
from Utils.variables import api, status_codes, num_gpus, launch_lock
from Utils.repositories import model_repository, forecast_repository, settings_repository, service_repository, historical_repository

trainers:dict[str, Trainer] = {}

@api.route("/train/<string:service_id>/<int:horizon>")
class Train(Resource):
    @api.doc(params={"service_id":"your-service-id"}, responses={200:"ok", 202:"working", 500:"Something ML died!!!!"})
    def post(self, service_id:str, horizon: int=12):
        launch_lock.acquire()
        services = service_repository.all()
        if not service_id in [str(service.id) for service in services]:
            return Response(status=400, response="Error, service doesn't exist")
        gpu_id = len(list(filter(lambda trainer: trainer._process.is_alive(), trainers.values()))) % num_gpus
        if not service_id in trainers:
            trainers[service_id] = Trainer(UUID(service_id), model_repository, forecast_repository, settings_repository, service_repository)
        elif trainers[service_id]._process.is_alive():
            return Response(status=202, response="Still working...")

        historical:list[Historical] = historical_repository.get_by_service(UUID(service_id))
        if not historical:
            trainers[service_id].forecaster.run(None, pd.to_timedelta(f"{horizon}s"))
            return Response(status=400, response=dumps({"message":f"Error, historical table is empty for service: {service_id}"}))


        trainers[service_id].run(historical[0], pd.to_timedelta(f"{horizon}s"), gpu_id)
        while not trainers[service_id]._process.is_alive(): pass # wait for trainer to actually have started

        launch_lock.release()
        return Response(status=200, response=dumps({"message":f"Training started for {service_id}"}))

@api.route("/train/<string:service_id>")
class TrainStatus(Resource):
    @api.doc(params={"service_id":"your-service-id"}, responses={code.status: str(res) for res, code in status_codes.items()})
    def get(self, service_id: str):
        return status_codes[trainers[service_id].is_idle()]

@api.route("/train/<string:service_id>/kill")
class TrainKill(Resource):

    @api.doc(params={"service_id": "your-service-id"}, responses={200:"killed", 500: "No trainers", 400:"no trainer present"})
    def get(self, service_id: str):
        if not service_id in trainers:
            return Response(status=500, response=f"error, no trainer in trainers for serviceid: {service_id}")
        if not trainers[service_id]._process.is_alive():
            return Response(status=400, response="Thread is already killed")

        trainers[service_id]._process.kill()

        return Response(status=200, response="killed")
