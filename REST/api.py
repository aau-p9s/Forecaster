from flask import Flask
from flask_restful import Resource, Api

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
        # Retrain model and predict + copy to DB
        pass

class Predict(Resource):
    def get(self):
        # Create new forecast and send back + copy to DB
        return {'new prediction': '2025-02-05 12:00:00,42'}
def start_api():
    app = Flask(__name__)
    api = Api(app)  
    api.add_resource(Forecast, '/forecast')
    api.add_resource(Train, '/train')
    api.add_resource(Predict, '/predict')
    app.run(debug=True)