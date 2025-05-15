from datetime import datetime
from time import time
from uuid import UUID
from darts import TimeSeries
from psycopg2 import DatabaseError
from Database.dbhandler import DbConnection
from Database.Utils import gen_uuid
from Database.Models.Forecast import Forecast


class ForecastRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def insert_forecast(self, forecast:Forecast, service_id:UUID) -> None:
        """Inserts a forecast linked to a model and a service."""
        query = 'INSERT INTO forecasts ("id", "modelid", "forecast", "serviceid", "createdat") VALUES (%s, %s, %s, %s, %s) RETURNING *;'
        serialForecast = forecast.serialize()
        params = [str(gen_uuid()), str(forecast.modelId), serialForecast, str(service_id), datetime.now()]
        self.db.execute(query, params)

    def upsert_forecast(self, forecast:Forecast, service_id:UUID) -> None:
        query = "INSERT INTO forecasts(id, modelid, forecast, serviceid, createdat) VALUES (%s, %s, %s, %s, %s) ON CONFLICT(id) DO UPDATE SET forecast = %s, createdat = %s"
        serialForecast = forecast.serialize()
        params = [str(gen_uuid()), str(forecast.modelId), serialForecast, str(service_id), datetime.now(), serialForecast, datetime.now()]
        self.db.execute(query, params)
