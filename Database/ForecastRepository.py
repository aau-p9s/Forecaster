from datetime import datetime
from Database.Entities.Forecast import Forecast
from Database.Repository import Repository
from Database.Utils import gen_uuid

class ForecastRepository(Repository[Forecast]):
    _class = Forecast
    def upsert_forecast(self, forecast:Forecast) -> None:
        query = "INSERT INTO forecasts(id, modelid, forecast, serviceid, createdat) VALUES (%s, %s, %s, %s, %s) ON CONFLICT(serviceid) DO UPDATE SET forecast = %s, createdat = %s"
        serialForecast = forecast.serialize()
        params = [str(gen_uuid()), str(forecast.model_id), serialForecast, str(forecast.service_id), datetime.now(), serialForecast, datetime.now()]
        self.db.execute(query, params)

    def table_name(self) -> str:
        return super().table_name() + "s"
