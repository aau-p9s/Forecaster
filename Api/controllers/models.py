from json import dumps
from flask import Response
from flask_restx import Resource
from ..lib.variables import api, model_repository

@api.route("/models")
class Models(Resource):
    @api.doc(responses={200:"ok"})
    def get(self):
        modelNames = map(str, model_repository.get_all_models())
        return Response(status=200, response=dumps({"message": "All models", "models":modelNames}))


