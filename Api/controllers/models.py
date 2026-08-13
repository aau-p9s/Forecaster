from json import dumps
from flask import Response
from flask_restx import Resource
from Utils.variables import api
from Utils.repositories import model_repository

@api.route("/models")
class Models(Resource):
    @api.doc(responses={200:"ok"})
    def get(self):
        modelNames = list(map(lambda model: model.name, model_repository.all()))
        return Response(status=200, response=dumps({"message": "All models", "models":modelNames}))


