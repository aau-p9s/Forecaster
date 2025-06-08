import os
from ML.Darts.Utils.load_model import load_model
from Utils.repositories import service_repository, model_repository
from Database.Entities.Model import Model
import warnings
warnings.filterwarnings("ignore")

prefix = "/var/model_tmp/BaselineModels"
models = [directory for directory in os.listdir(prefix) if os.path.isdir(f"{prefix}/{directory}")]
print(models)
data = {model:{} for model in models}
for model in models:
    print(model)
    files = os.listdir(f"{prefix}/{model}")
    for file in files:
        print(f"\t{file}")
        if file[-4:] == ".pth":
            with open(f"{prefix}/{model}/{file}", "rb") as file_reader:
                print("Saved model")
                data[model]["model"] = file_reader.read()
        if file[-9:] == ".pth.ckpt":
            with open(f"{prefix}/{model}/{file}", "rb") as file_reader:
                print("Saved ckpt")
                data[model]["ckpt"] = file_reader.read()
final_models = []
for model, byte_data in data.items():
    model_data = byte_data["model"]
    ckpt_data = byte_data["ckpt"] if "ckpt" in byte_data.keys() else None
    try:
        final_models.append((model, load_model(model, model_data, ckpt_data)))
    except:
        pass

services = service_repository.all()

model_repository.delete_all()
for service in services:
    for (name, model) in final_models:
        modelEntity = Model(name, service.id, model)
        model_repository.insert(modelEntity)
        print(f"Inserting {name} for {service.id}")

total_models = len(model_repository)
print(f"Total models: {total_models}")
