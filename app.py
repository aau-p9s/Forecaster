#!/usr/bin/env python3

#from Api.api import *
from ML.Darts.Tuning.Tuner import *
def insert_model():
    # model_path = "Assets\\autotheta_model.pth"
    # res = model_repository.insert_model("AutoThetaTest2", model_path)
    # print(f"Inserted: {res}")
    pass

if __name__ == '__main__':
    #start_api()

    series = load_data("Assets/ServerRequest1.csv", "min")
    series, missing_values_ratio = preprocessing.run_transformer_pipeline(series)
    tuner = Tuner("testId", series, 4)
    tuner.tune_model_x("RandomForest")