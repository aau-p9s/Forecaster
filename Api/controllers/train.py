from json import dumps
from multiprocessing import Process
from flask import Response
from flask_restx import Resource

from ML.Trainer import Trainer
from ..lib.variables import model_repository, api, trainers


@api.route("/train/<string:serviceId>")
class Train(Resource):
    @api.doc(params={"serviceId":"your-service-id"}, responses={200:"ok", 202:"working...", 500:"Something ML died!!!!"})
    def post(self, serviceId):
        # Retrain models on new thread and predict + copy to DB
        models = model_repository.get_all_models_by_service(serviceId)
        if not serviceId in trainers:
            trainer = Trainer(models, serviceId, None, None, None, model_repository)
            trainers[serviceId] = {
                "trainer": trainer
            }

        if trainers[serviceId]["thread"].is_alive():
            trainers[serviceId]["thread"].join()
            return Response(status=202, response=dumps({"message":f"Training was already in progress for {serviceId}"}))

        trainer:Trainer = trainers[serviceId]["trainer"]
        thread = Process(target=trainer.train_model)
        trainers[serviceId]["thread"] = thread
        thread.start()
        thread.join()

        return Response(status=200, response=dumps({"message":f"Training finished for {serviceId}"}))


