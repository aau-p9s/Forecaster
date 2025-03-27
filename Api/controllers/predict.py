from json import dumps
from multiprocessing import Process
from flask import Response
from flask_restx import Resource

from ML.Forecaster import Forecaster
from ..lib.variables import api, model_repository, forecasters, forecast_repository, settings_repository

@api.route("/predict/<string:serviceId>")
class Predict(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 500: "something died..."})
    def get(self, serviceId):
        # Create new forecast on a new thread and copy to DB
        models = model_repository.get_all_models_by_service(serviceId)
        if not serviceId in forecasters:
            forecaster = Forecaster(models, serviceId, forecast_repository)
            # TODO: use horizon from settings
            #settings = settings_repository.get_settings(serviceId)
            forecasters[serviceId] = {
                "forecaster":forecaster,
                "thread":Process(target=forecaster.create_forecasts, args=[12])
            }

        forecaster:Forecaster = forecasters[serviceId]["forecaster"]
        thread:Process = forecasters[serviceId]["thread"]

        # Check if an active forecasting thread exists for this service
        if not thread.is_alive():
            t = Process(target=forecaster.create_forecasts, args=[12])
            forecasters[serviceId]["thread"] = t
            t.start()
            t.join()
        else:
            thread.join()

        # TODO: implement logic
        # newest = Forecast(data[0], data[1])

        return Response(status=200, response=dumps({"message": f"Forecast finished for {serviceId}"}))#, "forecast":newest.forecast}))



