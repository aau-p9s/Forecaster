from json import dumps
from multiprocessing import Process
from uuid import UUID
from flask import Response
from flask_restx import Resource
from datetime import datetime

from Database.Models.Historical import Historical
from ML.Forecaster import Forecaster
from ..lib.variables import api, model_repository, forecast_repository, historical_repository, settings_repository, status_codes

forecasters:dict[str, Forecaster] = {}


@api.route("/predict/<string:service_id>/<int:forecast_horizon>")
class Predict(Resource):
    @api.doc(params={"service_id":"your-service-id"}, responses={200:"ok", 202:"working", 500: "something died..."})
    def post(self, service_id:str, forecast_horizon=12):
        historical:list[Historical] | None = historical_repository.get_by_service(UUID(service_id))
        if not historical:
            print("!!! WARNING !!! No data in historical table, this should not happen")

        if not service_id in forecasters:
            forecasters[service_id] = Forecaster(UUID(service_id), model_repository, forecast_repository, settings_repository)
        elif forecasters[service_id]._process.is_alive():
            return Response(status=202, response="Still working...")

        forecasters[service_id].predict(historical[0] if historical else None, forecast_horizon)


        return Response(status=200, response=dumps({"message": f"Forecasts finished for {service_id}"}))

    @api.doc(params={"service_id":"your-service-id"}, responses={code.status: str(res) for res, code in status_codes.items()})
    def get(self, service_id: str, forecast_horizon: int):
        return status_codes[forecasters[service_id]._process.is_alive()]

@api.route("/predict/<string:service_id>/kill")
class PredictKill(Resource):

    @api.doc(params={"service_id": "your-service-id"}, responses={200:"killed", 500:"no forecasters", 400:"no forecaster present"})
    def get(self, service_id: str):
        if not service_id in forecasters:
            return Response(status=500, response=f"error, no forecaster in forecasters for serviceid: {service_id}")
        if not forecasters[service_id]._process.is_alive():
            return Response(status=400, response="Thread is already killed")

        forecasters[service_id]._process.kill()

        return Response(status=200, response="killed")
