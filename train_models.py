from ML.Darts.Tuning.Tuner import *
from argparse import ArgumentParser
import os
import json
import darts.models as models
import numpy as np


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


parser = ArgumentParser(description="Train models")
parser.add_argument("-o", "--output", help="Output folder")
parser.add_argument("-i", "--input", type=file_path, help="Input data")

exclude_models = ["AutoARIMA", "ARIMA", "StatsforecastAutoARIMA"]

if __name__ == "__main__":
    parsed_args = parser.parse_args()
    input = parsed_args.input
    output = parsed_args.output
    series = load_data(input, "min")
    series, missing_values_ratio, _ = preprocessing.run_transformer_pipeline(
        series, resample=None
    )
    print(f"Min: {np.min(series.values())}, Max: {np.max(series.values())}\n")
    print(f"Ratio of missing values: {missing_values_ratio}\n")
    tuner = Tuner(
        "NoService",
        None,
        series,
        120,
        trials=75,
        output=output,
        exclude_models=exclude_models,
        dev_mode=True,
    )
    studies_and_models = tuner.tune_all_models()
