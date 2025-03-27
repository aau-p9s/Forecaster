from flask_restx import fields
from .variables import api

time_series_data = api.model("Time series model", {
    "timestamp": fields.DateTime,
    "value": fields.Float
})

tuning_model = api.model("Tuning POST Model", {
    "tuning_data": fields.Nested(time_series_data),
    "horizon":fields.Integer
})


