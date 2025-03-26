
from flask import Flask
from flask_restx import Api

from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.ServiceRepository import ServiceRepository
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
db = DbConnection(database, user, password, addr, port)
model_repository = ModelRepository(db)
forecast_repository = ForecastRepository(db)
service_repository = ServiceRepository(db)
trainer_threads = []
forecaster_threads = []
forecasters:dict[str, dict] = {}
trainers:dict[str, dict] = {}
