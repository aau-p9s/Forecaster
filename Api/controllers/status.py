
# quick endpoint to debug
from flask import Response
from flask_restx import Resource
from time import time
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
                format_model_status(name, status)
                for name, status in trainer.model_status.items()
            ])
            for id, trainer in trainers.items()
        ]))

def format_model_status(name: str, status:dict) -> str:
    start_time: float = status['start_time']
    end_time: float = status['end_time']
    message: str = status['message']
    error: str = status['error']
    time_str = str(end_time - start_time if end_time is not None else time() - start_time if start_time is not None else '')
    error_str = error if error is not None else ''
    return f"|\t{time_str:^7}\t|\t{name:^40}\t|\t{message:^7}\t|\t{error_str:^100}\t|"
