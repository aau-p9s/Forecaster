from flask import Flask
from flask_restx import Api, Resource
from Database.dbhandler import DbConnection

app = Flask(__name__)
api = Api(app, 
    version='1.0', 
    title='Sample API', 
    description='A simple Flask API with OpenAPI Spec'
)
db:DbConnection

@api.route("/forecast")
class Forecast(Resource):
    def get(self):
        return 1

def start_api(_db:DbConnection, addr:str, port:int):
    global db
    db = _db
    app.run(addr, port)
