from ML.Darts.Tuning.Tuner import *
from argparse import ArgumentParser
import os
import json

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = ArgumentParser(description="Train models")
parser.add_argument("-o", "--output", action="store_true", help="Output folder")
parser.add_argument("-i", "--input", type=dir_path, help="Input data")

if __name__ == '__main__':
    parsed_args = parser.parse_args()
    input = parsed_args.input
    series = load_data(input, "min")
    series, missing_values_ratio = preprocessing.run_transformer_pipeline(series)
    print(f"Ratio of missing values: {missing_values_ratio}\n")
    tuner = Tuner("NoServiceId", series, 10080, trials=1)
    studies_and_models = tuner.tune_all_models()

    for study, model in studies_and_models:
        print(f"Study: {study.study_name}, Model: {model.model_name}")

        output = parsed_args.output if parsed_args.output else "output"
        if not os.path.exists(output):
            os.makedirs(output)
        model_folder = os.path.join(output, model.name)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model.forecastingModel.save(f"{model_folder}/{model.name}.pth")
        with open(f"{model_folder}/best_trial.json", "w") as f:
            json.dump(study.best_trial, f, indent=4)
        print(f"Model and best trial data saved to {output}/{model.name}.pth")