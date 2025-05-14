from json import dumps
from multiprocessing import Process
from flask import Response
from flask_restx import Resource
from datetime import datetime

from ML.Forecaster import Forecaster
from ..lib.variables import api, model_repository, forecasters, forecast_repository, historical_repository

def format_data(data):
    return {
        "timestamp":[datetime.fromtimestamp(value[0]) for value in data["data"]["result"][0]["values"]],
        "value":[float(value[1]) for value in data["data"]["result"][0]["values"]]
    }

@api.route("/predict/<string:serviceId>/<forecastHorizon>")
class Predict(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 500: "something died..."})
    def get(self, serviceId, forecastHorizon=12):
        # Create new forecast on a new thread and copy to DB
        models = model_repository.get_all_models_by_service(serviceId)
        historical = historical_repository.get_by_service(serviceId)
        data = [format_data(obj.data) for obj in historical]
        if not serviceId in forecasters:
            forecaster = Forecaster(models, serviceId, forecast_repository, model_repository)
            # TODO: use horizon from settings
            forecasters[serviceId] = {
                "forecaster":forecaster,
                "thread":Process(target=forecaster.create_forecasts, args=[forecastHorizon, data[0]])
            }


        forecaster:Forecaster = forecasters[serviceId]["forecaster"]
        thread:Process = forecasters[serviceId]["thread"]

        # Check if an active forecasting thread exists for this service
        if not thread.is_alive():
            t = Process(target=forecaster.create_forecasts, args=[forecastHorizon, data[0]])
            forecasters[serviceId]["thread"] = t
            t.start()
            t.join()
        else:
            thread.join()

        

        return Response(status=200, response=dumps({"message": f"Forecasts finished for {serviceId}"}))#, "forecast":newest.forecast}))



