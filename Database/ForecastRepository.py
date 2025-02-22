from Database.dbhandler import DbConnection
from Database.Utils import gen_uuid


class ForecastRepository:
    def __init__(self, db: DbConnection):
        self.db = db

    def insert_forecast(self, model_id, forecast, serviceId):
        """Inserts a forecast linked to a model and a service."""
        query = 'INSERT INTO forecasts ("Id", "ModelId", "Forecast", "ServiceId") VALUES (%s, %s, %s, %s) RETURNING *;'
        params = (gen_uuid(), model_id, forecast, serviceId)
        return self.db.execute_query(query, params, fetch_one=True)

    def get_forecasts_by_model_and_service(self, model_id, serviceId):
        """Gets all forecasts for a given model and service."""
        query = 'SELECT * FROM forecasts WHERE "ModelId" = %s AND "ServiceId" = %s ORDER BY "Timestamp" ASC;'
        return self.db.execute_query(query, (model_id, serviceId))

    def get_forecasts_by_service(self, serviceId):
        """Gets all forecasts for a given model and service."""
        query = 'SELECT * FROM forecasts WHERE "ServiceId" = %s ORDER BY "Timestamp" ASC;'
        return self.db.execute_query(query, (serviceId,))

    def get_latest_forecast(self, model_id, serviceId):
        """Gets the latest forecast for a model and service."""
        query = 'SELECT * FROM forecasts WHERE "ModelId" = %s AND "ServiceId" = %s ORDER BY "Timestamp" DESC LIMIT 1;'
        return self.db.execute_query(query, (model_id, serviceId), fetch_one=True)

    def get_latest_forecast_by_service(self, serviceId):
        """Gets the latest forecast for a service."""
        query = 'SELECT * FROM forecasts WHERE "ServiceId" = %s ORDER BY "Timestamp" DESC LIMIT 1;'
        return self.db.execute_query(query, (serviceId,), fetch_one=True)