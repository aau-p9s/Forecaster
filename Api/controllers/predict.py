from json import dumps
from multiprocessing import Process
from uuid import UUID
from flask import Response
from flask_restx import Resource
from datetime import datetime

from Database.Models.Historical import Historical
from ML.Forecaster import Forecaster
from ..lib.variables import api, model_repository, forecast_repository, historical_repository


forecasters:dict[str, Forecaster] = {}

@api.route("/predict/<string:service_id>/<int:forecast_horizon>")
class Predict(Resource):
    @api.doc(params={"service_id":"your-service-id"}, responses={200:"ok", 500: "something died..."})
    def get(self, service_id:str, forecast_horizon=12):
        historical:list[Historical] | None = historical_repository.get_by_service(UUID(service_id))
        if not historical:
            print("!!! WARNING !!! No data in historical table, this should not happen")

        if not service_id in forecasters:
            forecasters[service_id] = Forecaster(UUID(service_id), model_repository, forecast_repository)

        forecasters[service_id].predict(historical if historical else None, forecast_horizon)


        return Response(status=200, response=dumps({"message": f"Forecasts finished for {service_id}"}))




