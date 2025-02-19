from flask import Flask, Response, json
from flask_restful import Resource, Api, reqparse
from Database.ForecastRepository import ForecastRepository
from Database.ModelRepository import ModelRepository
from ML.Trainer import Trainer
import threading
from threading import Thread
from Utils.ReturnableThread import ReturnableThread
from ML.Forecaster import Forecaster
from Database.dbhandler import DbConnection

trainer_threads = []
forecaster_threads = []
parser = reqparse.RequestParser()
parser.add_argument('modelname', type=str, help='Darts model name')
parser.add_argument('serviceid', type=str, help='Kubernetes service id')
db = DbConnection("p10s", "postgres", "password", "localhost", 5432)
model_repository = ModelRepository(db)
forecast_repository = ForecastRepository(db)

class Forecast(Resource):
    def get(self):
        args = parser.parse_args()
        serviceId = args['serviceid']
        forecast = forecast_repository.get_latest_forecast_by_service(serviceId)
        if all_threads_finished("forecaster", serviceId):
            return Response(status=200, response=json.dumps({"message": "Returned new and fresh forecast", "forecast": forecast}))
        else:
            return Response(status=202, response=json.dumps({"message": "Forecasting in progress. Returned archaic forecast", "forecast": forecast}))

class Train(Resource):
    def post(self):
        # Retrain models on new thread and predict + copy to DB
        args = parser.parse_args(strict=True)
        serviceId = args['serviceid']
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
            return Response(status=200, response=json.dumps({"message": f"All models trained for {serviceId}" }))
        else:
            return Response(status=202, response=json.dumps({"message": f"Training in progress for {serviceId}"}))
        
class Predict(Resource):
    def get(self):
        # Create new forecast on a new thread and copy to DB
        args = parser.parse_args(strict=True)
        serviceId = args["serviceid"]
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
                    return Response(status=200, response=json.dumps({"message": "Forecast created", "forecast": new_forecast}))
        # Forecast is not ready
        else:
            return Response(status=202, response=json.dumps({"message": "Forecast in progress"}))

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
    app = Flask(__name__)
    api = Api(app)  
    api.add_resource(Forecast, '/forecast')
    api.add_resource(Train, '/train')
    api.add_resource(Predict, '/predict')
    app.run(debug=True)

