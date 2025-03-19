from flask import Flask, Response
from flask_restx import Api, Resource
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.ServiceRepository import ServiceRepository
from Database.dbhandler import DbConnection
from json import dumps
from multiprocessing import Process
from ML.Forecaster import Forecast
from ML.Forecaster import Forecaster
from ML.Trainer import Trainer
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
api_port = getEnv("FORECASTER__PORT", "8081")
db = DbConnection(database, user, password, addr, port)
model_repository = ModelRepository(db)
forecast_repository = ForecastRepository(db)
service_repository = ServiceRepository(db)
trainer_threads = []
forecaster_threads = []
forecasters:dict[str, dict] = {}
trainers:dict[str, dict] = {}


@api.route("/train/<serviceId>")
class Train(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 202:"working...", 500:"Something ML died!!!!"})
    def post(self, serviceId):
        # Retrain models on new thread and predict + copy to DB
        models = model_repository.get_all_models_by_service(serviceId)
        if not serviceId in trainers:
            trainer = Trainer(models, serviceId, None, None, None, model_repository)
            trainers[serviceId] = {
                "trainer":trainer,
                "thread":Process(target=trainer.train_model)
            }
            trainers[serviceId]["thread"].start()
            return Response(status=200, response=dumps({"message":f"Trainer created and started for {serviceId}"}))

        trainer:Trainer = trainers[serviceId]["trainer"]
        thread:Process = trainers[serviceId]["thread"]

        if thread.is_alive():
            return Response(status=202, response=dumps({"message":f"Training is already in progress for {serviceId}"}))
        else:
            thread = Process(target=trainer.train_model)
            trainers[serviceId]["thread"] = thread
            thread.start()
            return Response(status=200, response=dumps({"message":f"Training started for {serviceId}"}))

@api.route("/predict/<serviceId>")
class Predict(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 202:"working...", 500: "something died..."})
    def get(self, serviceId):
        # Create new forecast on a new thread and copy to DB
        models = model_repository.get_all_models_by_service(serviceId)
        if not serviceId in forecasters:
            forecaster = Forecaster(models, serviceId, forecast_repository)
            forecasters[serviceId] = {
                "forecaster":forecaster,
                "thread":Process(target=forecaster.create_forecasts)
            }

        forecaster:Forecaster = forecasters[serviceId]["forecaster"]
        thread:Process = forecasters[serviceId]["thread"]

        # Check if an active forecasting thread exists for this service
        if not thread.is_alive():
            t = Process(target=forecaster.create_forecasts)
            forecasters[serviceId]["thread"] = t
            t.start()
            t.join()
        else:
            thread.join()

        data = forecast_repository.get_latest_forecast_by_service(forecaster.serviceId)
        newest = Forecast(data[0], data[1])
        
        return Response(status=200, response=dumps({"message":"Forecast finished", "forecast":newest.forecast}))


def start_api():
    app.run(api_addr, int(api_port), debug=True)

if __name__ == "__main__":
    app.run(api_addr, int(api_port))
