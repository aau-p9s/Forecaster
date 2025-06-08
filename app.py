#!/usr/bin/env python3
import sys
from Utils.getEnv import getEnv, help_dict
# BASICALLY A PREAMPLE THAT DISABLES A TON OF USELESS LOGGING
import warnings
enable_warnings = getEnv("FORECASTER__ENABLE__WARNINGS", "0") == "1"
if not enable_warnings:
    warnings.filterwarnings("ignore")
if "--help" in sys.argv:
    for key, value in help_dict.items():
        print(f"{key}:\t\tdefault: \t{value}")
# END CURSED STUFF
import torch
from Api.api import *
from Utils.variables import clear_data
from Utils.repositories import forecast_repository, historical_repository
from ML.Darts.Utils.models import PositiveGaussianLikelihood

if clear_data:
    forecast_repository.delete_all()
    historical_repository.delete_all()

if __name__ == '__main__':
    #torch.set_float32_matmul_precision('high')
    torch.set_num_threads(1)
    start_api()
