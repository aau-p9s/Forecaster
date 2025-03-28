from .lib.variables import api_addr, api_port, app
from .controllers import predict, train, tuner, models
from .lib import models
   


def start_api():
    app.run(api_addr, int(api_port), debug=True)

