
from multiprocessing import Lock
from flask import Flask, Response
from flask_restx import Api

from Database.ForecastRepository import ForecastRepository
from Database.HistoricalRepository import HistoricalRepository
from Database.ModelRepository import ModelRepository
from Database.ServiceRepository import ServiceRepository
from Database.SettingsRepository import SettingsRepository
from Database.dbhandler import DbConnection
from Utils.getEnv import getEnv


app = Flask(__name__)
api = Api(app, 
    version='1.0', 
    title='Forecaster API', 
    description='aau-p9s/Forecaster API service for machine learning predictive autoscaling'
)
database = getEnv("FORECASTER__PGSQL__DATABASE", "autoscaler")
user = getEnv("FORECASTER__PGSQL__USER", "root")
password = getEnv("FORECASTER__PGSQL__PASSWORD", "password")
addr = getEnv("FORECASTER__PGSQL__ADDR", "0.0.0.0")
port = getEnv("FORECASTER__PGSQL__PORT", "5432")
api_addr = getEnv("FORECASTER__ADDR", "0.0.0.0")
api_port = getEnv("FORECASTER__PORT", "8080")
train_timeout = int(getEnv("FORECASTER__TRAIN__TIMEOUT", "-1"))
num_gpus = int(getEnv("FORECASTER__TRAIN__GPUS", "2"))
db = DbConnection(database, user, password, addr, port)
launch_lock = Lock()
model_repository = ModelRepository(db)
forecast_repository = ForecastRepository(db)
service_repository = ServiceRepository(db)
settings_repository = SettingsRepository(db)
historical_repository = HistoricalRepository(db)
trainer_threads = []
forecaster_threads = []
forecasters:dict[str, dict] = {}
trainers:dict[str, dict] = {}
status_codes = {
    True: Response(status=202, response=str(True)),
    False: Response(status=200, response=str(False))
}
