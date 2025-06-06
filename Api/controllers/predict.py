from json import dumps
from uuid import UUID
from flask import Response
from flask_restx import Resource
import pandas as pd

from Database.Models.Historical import Historical
from ML.Forecaster import Forecaster
from ..lib.variables import api, model_repository, forecast_repository, historical_repository, settings_repository, service_repository, status_codes, num_gpus

forecasters:dict[str, Forecaster] = {}


@api.route("/predict/<string:service_id>/<int:horizon>")
class Predict(Resource):
    @api.doc(params={"service_id":"your-service-id"}, responses={200:"ok", 202:"working", 500: "something died..."})
    def post(self, service_id:str, horizon: int=12):
        services = service_repository.get_all_services()
        gpu_id = len(list(filter(lambda trainer: trainer._process.is_alive(), forecasters.values()))) % num_gpus
        if not service_id in [str(service.id) for service in services]:
            return Response(status=400, response="Error, service doesn't exist")
        historical:list[Historical] | None = historical_repository.get_by_service(UUID(service_id))
        if not historical:
            print("!!! WARNING !!! No data in historical table, this should not happen")

        if not service_id in forecasters:
            forecasters[service_id] = Forecaster(UUID(service_id), model_repository, forecast_repository, settings_repository)
        elif forecasters[service_id]._process.is_alive():
            return Response(status=202, response="Still working...")

        forecasters[service_id].run(historical[0] if historical else None, pd.to_timedelta(f"{horizon}s"), gpu_id)


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
