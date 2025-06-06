
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
        nl = "\n"
        t = "\t"
        return Response(status=200, response=f"""
            *** FORECASTERS ***
            {nl.join([
                f"{id}:{t}Status:{t}{'Working' if forecaster._process.is_alive() else 'Finished'}"
                for id, forecaster in forecasters.items()
            ])}
            *** TRAINERS ***
            {nl.join([
                f"{id}:{t}Status:{t}{'Working' if trainer._process.is_alive() else 'Finished'}"
                for id, trainer in trainers.items()
            ])}
        """)
