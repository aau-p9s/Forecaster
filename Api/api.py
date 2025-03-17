from flask import Flask, Response
from flask_restx import Api, Resource
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from Database.ServiceRepository import ServiceRepository
from Database.dbhandler import DbConnection
from json import dumps
from ML.forecaster import Forecaster
from ML.trainer import Trainer
from Utils.returnable_thread import ReturnableThread
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


@api.route("/train/<serviceId>")
class Train(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 202:"working...", 500:"Something ML died!!!!"})
    def post(self, serviceId):
        # Retrain models on new thread and predict + copy to DB
        models = model_repository.get_all_models_by_service(serviceId)

        # Check if an active forecasting thread exists for this service
        if not any(t.name == serviceId and t.is_alive() for t in trainer_threads):
            thread = ReturnableThread(target=Trainer(models, serviceId).train_model(), name=serviceId)
            trainer_threads.append(thread)
            thread.start()
        # If all threads (that should be 1 for now) for service are finished training, send to models DB
        if (all_threads_finished('trainer', serviceId)):
            finished_threads = get_finished_threads('trainer', serviceId)
            for t in finished_threads: # Allows for multi-threaded training for each model of a service in future (All models for a service are trained on one thread for now)
                for m in t.result:
                    model_repository.insert_model(m.name, m.binary, m.trainedTime, m.serviceId)
            return Response(status=200, response=dumps({"message": f"All models trained for {serviceId}" }))
        else:
            return Response(status=202, response=dumps({"message": f"Training in progress for {serviceId}"}))


@api.route("/predict/<serviceId>")
class Predict(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 202:"working...", 500: "something died..."})
    def get(self, serviceId):
        # Create new forecast on a new thread and copy to DB
        models = model_repository.get_all_models_by_service(serviceId)

        # Check if an active forecasting thread exists for this service
        if not any(t.name == serviceId and t.is_alive() for t in forecaster_threads):
            thread = ReturnableThread(target=Forecaster(models, serviceId).create_forecasts(historical_data=None), name=serviceId)
            forecaster_threads.append(thread)
            thread.start()

        # Forecast is ready
        if all_threads_finished("forecaster", serviceId):
            finished_threads = get_finished_threads('forecaster', serviceId)
            for t in finished_threads: # One thread for each service
                for f in t.result: # Allows for multi-threaded forecasting for each model of a service in future (Only one forecast is returned for now)
                    forecast_repository.insert_forecast(f.modelId, f.forecast, serviceId)
                    new_forecast = f.forecast
                    return Response(status=200, response=dumps({"message": "Forecast created", "forecast": new_forecast}))
        # Forecast is not ready
        else:
            return Response(status=202, response=dumps({"message": "Forecast in progress"}))

def get_running_threads(type, serviceId):
    """Get running threads for a given type and serviceId.
    Args:
      type (str): "forecaster" or "trainer"
      serviceId (str): The specific service ID to filter.
    Returns:
      list: Active threads of the specified type and serviceId.
    """
    if type == "trainer":
        return [t for t in trainer_threads if t.is_alive() and t.name == serviceId]
    elif type == "forecaster":
        return [t for t in forecaster_threads if t.is_alive() and t.name == serviceId]
    else:
        raise ValueError("Invalid thread type. Use 'forecaster' or 'trainer'.")

def all_threads_finished(type, serviceId):
    """Check if all threads with a given serviceId have finished.
    Args:
      type (str): "forecaster" or "trainer"
      serviceId (str): The specific service ID to check.
    Returns:
      bool: Whether all threads for the given serviceId have finished.
    """
    if type == "trainer":
        return all(not t.is_alive() for t in trainer_threads if t.name == serviceId)
    elif type == "forecaster":
        return all(not t.is_alive() for t in forecaster_threads if t.name == serviceId)
    else:
        raise ValueError("Invalid thread type. Use 'forecaster' or 'trainer'.")
    
def get_finished_threads(type, serviceId):
    """Get all finished threads with a given serviceId.
    
    Args:
      type (str): "forecaster" or "trainer"
      serviceId (str): The specific service ID to check.

    Returns:
      list: A list of finished threads matching the given serviceId.
    """
    if type == "trainer":
        return [t for t in trainer_threads if t.name == serviceId and not t.is_alive()]
    elif type == "forecaster":
        return [t for t in forecaster_threads if t.name == serviceId and not t.is_alive()]
    else:
        raise ValueError("Invalid thread type. Use 'forecaster' or 'trainer'.")
  


def start_api():

    app.run(api_addr, api_port, debug=True)

if __name__ == "__main__":
    app.run(api_addr, api_port)
