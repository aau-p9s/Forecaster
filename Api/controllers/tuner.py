from json import dumps
from flask import Response, request
from flask_restx import Resource
from uuid import UUID
from Database.Models.Model import Model
from Database.Utils import gen_uuid
from ML.Darts.Tuning.Tuner import Tuner
from ML.Darts.Utils import preprocessing
from ..lib.models import tuning_model
from ..lib.variables import api
from ..lib.variables import model_repository, forecasters, historical_repository
from Database.Models.Historical import Historical

@api.route("/tuner/<string:serviceId>/<string:modelId>")
class Tune(Resource):
    @api.doc(params={"serviceId": "your-service-id", "modelId":"internal-model-id"}, responses={200:"ok", 202:"working...", 500: "something died..."})
    @api.expect(tuning_model)
    def post(self, serviceId:UUID, modelId:UUID, forecast_horizon=12):
        data = request.get_json()
        forecast_horizon = data["horizon"]
        try:
            tuningData = preprocessing.load_json_data(data)
        except Exception as e:
            raise ValueError(f"Time series could not be processed {str(e)}")

        tuningData, missing_values_ratio, _ = preprocessing.run_transformer_pipeline(tuningData)
        try:
            model = model_repository.get_by_modelid_and_service(modelId, serviceId)
            if model is None:
                return Response(status=404, response=dumps({"message": "Model not found."}))
        except Exception as e:
            return Response(status=500, response=dumps({"message": "Error fetching model.", "error": str(e)}))

        t = Tuner(serviceId, model_repository, tuningData, forecast_horizon)
        
        complete_study, tuned_model = t.tune_model_x(model)
        if complete_study["value"] == "Infinity":
            print("Best value was inf, which means an error probably occured")
            return Response(status=400, response=dumps({"message": "Model tuned, but not saved "}))
        model_repository.upsert_model(tuned_model)
        
        return Response(status=200, response=dumps({"message": "Model tuned. Study returned.", "study": complete_study}))

@api.route("/tuner/<string:serviceId>")
class TuneAll(Resource):
    @api.doc(responses={200:"ok", 202:"working...", 500: "something died..."})
    # TODO: update this data to be correct
    @api.expect(tuning_model)
    def post(self, serviceId:UUID, forecast_horizon=12):
        data = request.get_json() 
        forecast_horizon = data["horizon"]
        tuningData = preprocessing.load_json_data(data)
        tuningData, missing_values_ratio, _ = preprocessing.run_transformer_pipeline(tuningData)

        t = Tuner(serviceId, model_repository, tuningData, forecast_horizon)
        try:
            studies_and_models = t.tune_all_models()
        except Exception as e:
            return Response(status=500, response=dumps({"message": "Error tuning models.", "error": str(e)}))
        return Response(status=200, response=dumps({"message": "All models tuned. Study returned.", "study": studies_and_models[0]}))
 
