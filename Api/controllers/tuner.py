from json import dumps
from flask import Response, request
from flask_restx import Resource

from ML.Darts.Tuning.Tuner import Tuner
from ..lib.models import tuning_model
from ..lib.variables import api


@api.route("/tuner/<string:serviceId>/<string:modelName>")
class Tune(Resource):
    @api.doc(params={"modelName":"darts-model-name"}, responses={200:"ok", 202:"working...", 500: "something died..."})
    @api.expect(tuning_model)
    def get(self, serviceId, modelName):
        data = request.get_json()
        tuningData = data["tuningData"]
        forecast_horizon = data["horizon"]

        t = Tuner(serviceId, tuningData, forecast_horizon)
        complete_study = t.tune_model_x(modelName)
        return Response(status=200, response=dumps({"message": "Model tuned. Study returned.", "study": complete_study}))

@api.route("/tuner/<string:serviceId>")
class TuneAll(Resource):
    @api.doc(responses={200:"ok", 202:"working...", 500: "something died..."})
    # TODO: update this data to be correct
    @api.expect(tuning_model)
    def post(self, serviceId):
        data = request.get_json()
        tuningData = data["tuningData"]
        forecast_horizon = data["horizon"]

        t = Tuner(serviceId, tuningData, forecast_horizon)
        complete_study = t.tune_all_models()
        return Response(status=200, response=dumps({"message": "Model tuned. Study returned.", "study": complete_study}))
 
