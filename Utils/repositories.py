from Database.ForecastRepository import ForecastRepository
from Database.HistoricDataRepository import HistoricDataRepository
from Database.ModelRepository import ModelRepository
from Database.ServiceRepository import ServiceRepository
from Database.SettingsRepository import SettingsRepository
from Database.dbhandler import DbConnection
from Utils.variables import database, user, password, addr, port

db = DbConnection(database, user, password, addr, port)
model_repository = ModelRepository(db)
forecast_repository = ForecastRepository(db)
service_repository = ServiceRepository(db)
settings_repository = SettingsRepository(db)
historical_repository = HistoricDataRepository(db)
