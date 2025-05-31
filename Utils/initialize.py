from os import listdir, remove
from uuid import uuid4
from datetime import datetime
import traceback


def gen_uuid() -> str:
    return str(uuid4())


def main(models, connection):
    loaded_models = {}

    # insert a dummy workload
    #cursor = connection.cursor()
    #cursor.execute("INSERT INTO services(Id,Name,AutoscalingEnabled) VALUES (%s, %s, %s)", [
    #    gen_uuid(),
    #    "workload-api-deployment",
    #    False
    #])
    
    cursor = connection.cursor()
    cursor.execute("SELECT id FROM services")
    serviceIds = cursor.fetchall()

    tab = "\t"
    
    for model in models:
        print(f"{model.__name__} ... ", end=f"\r{tab*8}")
        try:
            model.load(f"./models/{model.__name__}/{model.__name__}.pth")
            print("\033[38;2;0;255;0msuccess!\033[0m")
            loaded_models[model.__name__] = model
        except Exception as e:
            print(f"\033[38;2;255;0;0mfailed!\033[0m {e=}")

    for model in loaded_models.values():
        with open(f"./models/{model.__name__}/{model.__name__}.pth", "rb") as file:
            data = file.read()
        if f"{model.__name__}.pth.ckpt" in listdir(f"./models/{model.__name__}"):
            with open(f"./models/{model.__name__}/{model.__name__}.pth.ckpt", "rb") as file:
                ckpt = file.read()
        else:
            ckpt = None
        cursor = connection.cursor()
        for serviceId in serviceIds:
            cursor.execute(f"INSERT INTO models(Id,Name,ServiceId,Bin,TrainedAt,Ckpt) VALUES(%s, %s, %s, %s, %s, %s)", [
                gen_uuid(),
                model.__name__,
                serviceId,
                data,
                datetime.now(),
                ckpt
            ])
        connection.commit()

    print("Validating...")
    cursor = connection.cursor()
    cursor.execute("SELECT Name, ServiceId, Bin, Ckpt FROM models")
    rows = cursor.fetchall()
    for name, id, data, ckpt in rows:
        print(f"{id} - {name} ...", end=f"\r{tab*8}")
        with open(f"/tmp/{name}.pth", "wb") as file:
            file.write(data)
        if ckpt is not None:
            with open(f"/tmp/{name}.pth.ckpt", "wb") as file1:
                file1.write(ckpt)

        try:
            loaded_models[name].load(file.name)
            print("\033[38;2;0;255;0msuccess!\033[0m")
        except Exception as e:
            print("\033[38;2;255;0;0mfailed!\033[0m {e=}")
            traceback.print_exc()

        remove(file.name)
