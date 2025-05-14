from json import dumps, loads
from multiprocessing import Process
from flask import Response
from flask_restx import Resource
from datetime import datetime

from Database.Models.Historical import Historical
from ML.Trainer import Trainer
from ..lib.variables import model_repository, api, forecast_repository, historical_repository

trainers:dict[str, Trainer] = {}

@api.route("/train/<string:service_id>/<int:forecast_horizon>")
class Train(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 202:"working...", 500:"Something ML died!!!!"})
    def post(self, service_id, forecast_horizon=12):
        historical:list[Historical] = historical_repository.get_by_service(service_id)
        if not historical:
            return Response(status=400, response=dumps({"message":f"Error, historical table is empty for service: {service_id}"}))

        if not service_id in trainers:
            trainers[service_id] = Trainer(service_id, model_repository, forecast_repository)

        trainers[service_id].train(historical[0], forecast_horizon)

        return Response(status=200, response=dumps({"message":f"Training started for {service_id}"}))


