from ML.Darts.Tuning.Tuner import *
from argparse import ArgumentParser
import os
import json
import darts.models as models


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError


exclude_models = [
    "AutoARIMA",
    "ARIMA",
    "KalmanForecaster",
    "GlobalNaiveAggregate",
    "NaiveDrift",
    "NaiveMean",
    "NaiveMovingAverage",
    "NaiveSeasonal",
    "VARIMA",
    "RegressionEnsembleModel",
    "NaiveEnsembleModel",
    "EnsembleModel",
    "BlockRNNModel",
]

considered_models = [
    models.StatsForecastAutoCES,
]

parser = ArgumentParser(description="Train models")
parser.add_argument("-o", "--output", help="Output folder")
parser.add_argument("-i", "--input", type=file_path, help="Input data")

if __name__ == "__main__":
    parsed_args = parser.parse_args()
    input = parsed_args.input
    output = parsed_args.output
    series = load_data(input, "h")
    series, missing_values_ratio = preprocessing.handle_missing_values(series)
    # series, missing_values_ratio = preprocessing.run_transformer_pipeline(series)
    print(f"Ratio of missing values: {missing_values_ratio}\n")
    tuner = Tuner(
        "NoServiceId",
        series,
        168,
        trials=1,
        exclude_models=exclude_models,
        gpu=1,
        output=output,
    )

    for model in considered_models:
        studies_and_models = tuner.tune_model_x(model)
