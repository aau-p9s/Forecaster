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

    series = load_data("ML\Darts\Assets\ServerRequest1.csv", "min")
    series, missing_values_ratio = preprocessing.run_transformer_pipeline(series)
    print(f"Ratio of missing values: {missing_values_ratio}\n")
    tuner = Tuner("testId", series, 4, trials=30)
    #with open("Assets/autotheta_model.pth", "rb") as bin:
    #    model = Model("testModelId", "AutoTheta", bin.read(), "testId")
    tuner.tune_model_x("RNNModel")