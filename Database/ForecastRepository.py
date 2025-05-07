from time import time
from darts import TimeSeries
from psycopg2 import DatabaseError
import psycopg2
from Database.dbhandler import DbConnection
from Database.Utils import gen_uuid
from Database.Models.Forecast import Forecast


class ForecastRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def insert_forecast(self, forecast:Forecast, service_id) -> None:
        """Inserts a forecast linked to a model and a service."""
        query = 'INSERT INTO forecasts ("id", "modelid", "forecast", "serviceid", "createdat") VALUES (%s, %s, %s, %s, %s) RETURNING *;'
        serialForecast = forecast.serialize()
        params = [gen_uuid(), forecast.modelId, serialForecast, service_id, psycopg2.TimestampFromTicks(time())]
        self.db.execute(query, params)

