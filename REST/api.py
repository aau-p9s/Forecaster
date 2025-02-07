from flask import Flask
from flask_restful import Resource, Api, reqparse
from ML.Trainer import Trainer
from threading import Thread
from ML.Forecaster import Forecaster
from Database.dbhandler import DbConnection

trainer_threads = []
forecaster_threads = []
parser = reqparse.RequestParser()
parser.add_argument('modelname', type=str, help='Darts model name')
db = DbConnection("p10s", "postgres", "password", "localhost", 5432)

class Forecast(Resource):
    def get(self):
        return {'forecast': '''
                2025-02-05 12:00:00,120
                2025-02-05 12:01:00,135
                2025-02-05 12:02:00,128
                2025-02-05 12:03:00,140
                2025-02-05 12:04:00,150
                2025-02-05 12:05:00,145
                2025-02-05 12:06:00,160
                2025-02-05 12:07:00,155
                '''}
    
class Train(Resource):
    def post(self):
        # Retrain model on new thread and predict + copy to DB
        args = parser.parse_args(strict=True)
        model = db.get_by_model_name(args['modelname'])
        print(model)
        # trainer_threads.append(Thread(target=Trainer()))
        

class Predict(Resource):
    def get(self):
        # Create new forecast on a new thread (increase in no. of Kube services may cause this to bottleneck the app) and send back + copy to DB
        
        return {'new prediction': '2025-02-05 12:00:00,42'}
def start_api():
    app = Flask(__name__)
    api = Api(app)  
    api.add_resource(Forecast, '/forecast')
    api.add_resource(Train, '/train')
    api.add_resource(Predict, '/predict')
    app.run(debug=True)
