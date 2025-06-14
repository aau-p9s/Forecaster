
# quick endpoint to debug
from datetime import datetime
from flask import Response
from flask_restx import Resource
from time import time
from Utils.variables import api
from Api.controllers.predict import forecasters
from Api.controllers.train import trainers

start_time = datetime.now()

@api.route("/status")
class Predict(Resource):
    @api.doc(responses={200:"ok"})
    def get(self):
        process_status = "\n".join([
            "-"*30,
            "*** FORECASTERS ***"
        ] + [
            f"{id}:\tStatus:\t{forecaster.status.get()} ({forecaster.finished.get()}/{forecaster.total.get()})"
            for id, forecaster in forecasters.items()
        ] + [
            "-"*30,
            "*** TRAINERS ***"
        ] + [
            f"{id}:\tStatus:\t{trainer.status.get()}"
            for id, trainer in trainers.items()
        ])

        model_tables = {id:[["Running time", "Model name", "Status", "Error"]] + [
                format_model_status(name, status)
                for name, status in trainer.model_status.items()
            ]
            for id, trainer in trainers.items()
        }

        final_model_tables = {
            id: str_table(format_table(table))
            for id, table in model_tables.items()
        }
        return Response(status=200, response="\n".join([f"Running time: {datetime.now() - start_time}"] + [process_status] + ["-"*30]+ [f"Model status for {id}:\n{table}" for id, table in final_model_tables.items()]))





def format_model_status(name: str, status:dict) -> list[str]:
    start_time: float = status['start_time']
    end_time: float = status['end_time']
    message: str = status['message']
    error: str = status['error']
    time_str = f"{end_time - start_time if end_time is not None else time() - start_time if start_time is not None else 0.00:.2f}"
    error_str = error if error is not None else ''
    return [time_str, name, message, error_str]

def format_table(table: list[list[str]]):
    final_table = []
    column_max_widths = []
    if not table:
        return table
    for column_index in range(len(table[0])):
        column_max_widths.append(max([len(row[column_index]) for row in table]))
    for row in table:
        final_table.append([value + (" " * (width - len(value))) for width, value in zip(column_max_widths, row)])
    return final_table

def str_table(table: list[list[str]]):
    return "\n".join([
        f"| {' | '.join(row)} |"
        for row in table
    ])
