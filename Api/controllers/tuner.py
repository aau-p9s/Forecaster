from json import dumps
from flask import Response, request
from flask_restx import Resource

from Database.Models.Model import Model
from Database.Utils import gen_uuid
from ML.Darts.Tuning.Tuner import Tuner
from ML.Darts.Utils import preprocessing
from ..lib.models import tuning_model
from Utils.variables import model_repository, forecasters, forecast_repository, historical_repository, api


@api.route("/tuner/<string:serviceId>/<string:modelId>")
class Tune(Resource):
    @api.doc(params={"modelId":"internal-model-id"}, responses={200:"ok", 202:"working...", 500: "something died..."})
    @api.expect(tuning_model)
    def post(self, serviceId, modelId):
        data = request.get_json()
        tuningData = data["tuning_data"]
        forecast_horizon = data["horizon"]

        tuningData = preprocessing.load_json_data(tuningData)
        tuningData, missing_values_ratio, _ = preprocessing.run_transformer_pipeline(tuningData)
        try:
            model = model_repository.get_by_modelid_and_service(modelId, serviceId)
        except Exception as e:
            return Response(status=500, response=dumps({"message": "Error fetching model.", "error": str(e)}))

        if model is None:
            return Response(status=404, response=dumps({"message": "Model not found."}))

        t = Tuner(serviceId, tuningData, forecast_horizon)
        complete_study, trained_model = t.tune_model_x(model)
        model_repository.insert_model(trained_model)
        
        return Response(status=200, response=dumps({"message": "Model tuned. Study returned.", "study": complete_study}))

@api.route("/tuner/<string:serviceId>")
class TuneAll(Resource):
    @api.doc(responses={200:"ok", 202:"working...", 500: "something died..."})
    # TODO: update this data to be correct
    @api.expect(tuning_model)
    def post(self, serviceId):
        data = request.get_json()
        tuningData = data["tuning_data"]
        forecast_horizon = data["horizon"]

        tuningData = preprocessing.load_json_data(tuningData)
        tuningData, missing_values_ratio, _ = preprocessing.run_transformer_pipeline(tuningData)

        t = Tuner(serviceId, model_repository, tuningData, forecast_horizon)
        try:
            studies_and_models = t.tune_all_models()
        except Exception as e:
            return Response(status=500, response=dumps({"message": "Error tuning models.", "error": str(e)}))
        return Response(status=200, response=dumps({"message": "All models tuned. Study returned.", "study": studies_and_models[0]}))
 
