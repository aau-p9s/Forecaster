
# quick endpoint to debug
from flask import Response
from flask_restx import Resource
from Api.lib.variables import api
from Api.controllers.predict import forecasters
from Api.controllers.train import trainers

@api.route("/status")
class Predict(Resource):
    @api.doc(responses={200:"ok"})
    def get(self):
        return Response(status=200, response="\n".join([
            "*** FORECASTERS ***"
        ] + [
            f"{id}:\tStatus:\t{'Working' if forecaster._process.is_alive() else 'Finished'}"
            for id, forecaster in forecasters.items()
        ] + [
            "*** TRAINERS ***"
        ] + [
            f"{id}:\tStatus:\t{'Working' if trainer._process.is_alive() else 'Finished'}\n" +
            "\n".join([
                f"\t{name}:\r\t\t\t\t\t{status}"
                for name, status in trainer.model_status.items()
            ])
            for id, trainer in trainers.items()
        ]))
