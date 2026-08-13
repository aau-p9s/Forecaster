from Utils.variables import api_addr, api_port, app
from .controllers import predict, train, models, status

def start_api():
    app.run(api_addr, int(api_port), debug=True)

