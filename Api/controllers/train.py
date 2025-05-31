from json import dumps, loads
from multiprocessing import Process
from uuid import UUID
from flask import Response
from flask_restx import Resource
from datetime import datetime

from Database.Models.Historical import Historical
from ML.Trainer import Trainer
from ..lib.variables import model_repository, api, forecast_repository, historical_repository, settings_repository, status_codes

trainers:dict[str, Trainer] = {}

@api.route("/train/<string:service_id>/<int:forecast_horizon>")
class Train(Resource):
    @api.doc(params={"service_id":"your-service-id"}, responses={200:"ok", 202:"working", 500:"Something ML died!!!!"})
    def post(self, service_id:str, forecast_horizon=12):
        if not service_id in trainers:
            trainers[service_id] = Trainer(UUID(service_id), model_repository, forecast_repository, settings_repository)
        elif trainers[service_id]._process.is_alive():
            return Response(status=202, response="Still working...")

        historical:list[Historical] = historical_repository.get_by_service(UUID(service_id))
        if not historical:
            trainers[service_id].forecaster.predict(None, forecast_horizon)
            return Response(status=400, response=dumps({"message":f"Error, historical table is empty for service: {service_id}"}))


        trainers[service_id].train(historical[0], forecast_horizon)

        return Response(status=200, response=dumps({"message":f"Training started for {service_id}"}))

    @api.doc(params={"service_id":"your-service-id"}, responses={code.status: res for res, code in status_codes.items()})
    def get(self, service_id: str, forecast_horizon: int):
        return status_codes[trainers[service_id]._process.is_alive()]


@api.route("/train/<string:service_id>/kill")
class TrainKill(Resource):

    @api.doc(params={"service_id": "your-service-id"}, responses={200:"killed", 400:"no trainer present"})
    def get(self, service_id: str):
        if not service_id in trainers:
            return Response(status=500, response=f"error, no trainer in trainers for serviceid: {service_id}")
        if not trainers[service_id]._process.is_alive():
            return Response(status=400, response="Thread is already killed")

        trainers[service_id]._process.kill()

        return Response(status=200, response="killed")
