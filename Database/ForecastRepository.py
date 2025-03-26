from psycopg2 import DatabaseError
from Database.dbhandler import DbConnection
from Database.Utils import gen_uuid
from Database.Models.Forecast import Forecast


class ForecastRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def insert_forecast(self, model_id, forecast, serviceId) -> None:
        """Inserts a forecast linked to a model and a service."""
        query = 'INSERT INTO forecasts ("id", "modelid", "forecast", "serviceid") VALUES (%s, %s, %s, %s) RETURNING *;'
        params = (gen_uuid(), model_id, forecast, serviceId)
        self.db.execute_query(query, params)

    def get_forecasts_by_model_and_service(self, model_id, serviceId) -> list[Forecast]:
        """Gets all forecasts for a given model and service."""
        query = 'SELECT modelid, forecast FROM forecasts WHERE "modelid" = %s AND "serviceid" = %s ORDER BY "createdat" ASC;'
        rows = self.db.execute_query(query, [model_id, serviceId])
        return [Forecast(row[0], row[1], 0.0) for row in rows]

    def get_forecasts_by_service(self, serviceId) -> list[Forecast]:
        """Gets all forecasts for a given model and service."""
        query = 'SELECT modelid, forecast FROM forecasts WHERE "serviceid" = %s ORDER BY "createdat" ASC;'
        rows = self.db.execute_query(query, (serviceId,))
        return [Forecast(row[0], row[1], 0.0) for row in rows]

    def get_latest_forecast(self, model_id, serviceId) -> Forecast:
        """Gets the latest forecast for a model and service."""
        query = 'SELECT modelid, forecast FROM forecasts WHERE "modelid" = %s AND "serviceid" = %s ORDER BY "createdat" DESC LIMIT 1;'
        rows = self.db.execute_query(query, (model_id, serviceId))
        if len(rows) > 0:
            row = rows[0]
            return Forecast(row[0], row[1], 0.0)
        raise DatabaseError

    def get_latest_forecast_by_service(self, serviceId) -> Forecast:
        """Gets the latest forecast for a service."""
        query = 'SELECT modelid, forecast FROM forecasts WHERE "serviceid" = %s ORDER BY "createdat" DESC LIMIT 1;'
        rows = self.db.execute_query(query, (serviceId,))
        if len(rows) > 0:
            row = rows[0]
            return Forecast(row[0], row[1], 0.0)
        raise DatabaseError
