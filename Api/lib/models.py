from flask_restx import fields
from .variables import api

time_series_data = api.model("TimeSeriesData", {
    "timestamp": fields.List(fields.DateTime(dt_format='iso8601')),
    "value": fields.List(fields.Float)
})

tuning_model = api.model("TuningPostModel", {
    "tuning_data": fields.Nested(time_series_data),
    "horizon": fields.Integer
})