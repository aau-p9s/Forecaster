import datetime

class Model:
    def __init__(self, name, binary, serviceId):
        self.name = name
        self.binary = binary
        self.trainedTime = datetime.date.today()
        self.serviceId = serviceId