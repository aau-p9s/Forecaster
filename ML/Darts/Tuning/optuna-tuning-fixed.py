#!/usr/bin/env python
# coding: utf-8

import torch
import json
import os
from darts.models import RNNModel, BlockRNNModel, TFTModel, NBEATSModel, NHiTSModel, TCNModel, TransformerModel, DLinearModel, NLinearModel, TiDEModel, TSMixerModel
from darts.utils.likelihood_models import GaussianLikelihood
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
import optuna
import json
from darts.utils.likelihood_models import GaussianLikelihood
from sklearn.metrics import root_mean_squared_error
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping, Callback
import numpy as np
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Set the precision to medium for faster matmul operations

torch.set_float32_matmul_precision("medium")

# Load data into a TimeSeries object


df = pd.read_csv("agg_minute.csv")  # Tuesday ->
df["time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")
series1 = TimeSeries.from_dataframe(
    df, time_col="time", value_cols="count", freq="min"
)  # Adjust frequency

series = series1.astype(np.float32)

df = pd.read_csv("agg_daily.csv")  # Update the path to your dataset
df["time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d")
series_daily = TimeSeries.from_dataframe(
    df, time_col="time", value_cols="count", freq="d"
)  # Adjust frequency

df = pd.read_csv("agg_second.csv")  # Update the path to your dataset
df["time"] = pd.to_datetime(df["date_time"], format="%Y-%m-%d %H:%M:%S")
series_secondly = TimeSeries.from_dataframe(
    df, time_col="time", value_cols="count", freq="s"
)  # Adjust frequency

series_daily = series_daily.astype(np.float32)
series_secondly = series_secondly.astype(np.float32)


scaler_min = Scaler()
series_min_scaled = scaler_min.fit_transform(series)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    num_workers = 8
else:
    num_workers = 0

# Split the dataset into training and validation sets (6 day training, 1 day forecast)
ln = 1440 * 4  # 4 days for training
last_days = series_min_scaled[-2880:]
train_min, val_min = series_min_scaled[:ln], last_days


class PositiveGaussianLikelihood(GaussianLikelihood):
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        
        # Ensure that std is always non-negative before taking the exp
        if torch.any(result.std < 0):
            result.std = torch.abs(result.std)  # Take absolute value to prevent negative std
        result.std = torch.exp(result.std)  # Ensures a positive std after exp
        return result


def tune_model(model_name, trials):
    best_model = None
    best_score = float("inf")

    def objective(trial):
        global best_model, best_score
        my_stopper = EarlyStopping(
            monitor="train_loss",
            patience=5,
            min_delta=0.05,
            mode="min",
        )

        pl_trainer_kwargs = {
            "callbacks": [my_stopper],
            "accelerator": "gpu",
            "devices": [0],
        }
        # Define hyperparams search space:
        input_chunk_length = trial.suggest_int("input_chunk_length", 50, 1440)
        output_chunk_length = trial.suggest_int("output_chunk_length", 1, 24)
        hidden_dim = trial.suggest_int("hidden_dim", 10, 100)
        dropout = trial.suggest_float("dropout", 0.1, 0.4) #This was (0.0, 0.4) for all the others
        n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        n_epochs = trial.suggest_int("n_epochs", 10, 100)
        training_length = trial.suggest_int(
            "training_length", 1440, 2880
        )  # 1 day - 2 days
        lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
        num_blocks = trial.suggest_int("num_blocks", 1, 3)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        layer_widths = trial.suggest_int("layer_widths", 32, 256)

        # PyTorch (Lightning)-based Models (GlobalForecastingModel)
        rnn_model = RNNModel(
            model="RNN",
            input_chunk_length=input_chunk_length,
            # output_chunk_length=output_chunk_length, # This is fixed at 1
            hidden_dim=hidden_dim,
            dropout=dropout,
            optimizer_kwargs={"lr": lr},
            n_rnn_layers=n_rnn_layers,
            batch_size=batch_size,
            n_epochs=n_epochs,
            training_length=training_length,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        lstm_model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            # output_chunk_length=output_chunk_length, # This is fixed at 1
            hidden_dim=hidden_dim,
            dropout=dropout,
            optimizer_kwargs={"lr": lr},
            n_rnn_layers=n_rnn_layers,
            batch_size=batch_size,
            n_epochs=n_epochs,
            training_length=training_length,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        gru_model = RNNModel(
            model="GRU",
            input_chunk_length=input_chunk_length,
            # output_chunk_length=output_chunk_length, # This is fixed at 1
            hidden_dim=hidden_dim,
            dropout=dropout,
            optimizer_kwargs={"lr": lr},
            n_rnn_layers=n_rnn_layers,
            batch_size=batch_size,
            n_epochs=n_epochs,
            training_length=training_length,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        tft_model = TFTModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            add_relative_index=True,
            hidden_size=trial.suggest_int("hidden_size", 16, 50, log=True),
            lstm_layers=n_rnn_layers,
            dropout=dropout,
            batch_size=batch_size,
            optimizer_kwargs={"lr": lr},
            n_epochs=n_epochs,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        blockrnn_RNNmodel = BlockRNNModel(
            model="RNN",
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_rnn_layers=n_rnn_layers,
            n_epochs=n_epochs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            optimizer_kwargs={"lr": lr},
            batch_size=batch_size,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        blockrnn_LSTMmodel = BlockRNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_rnn_layers=n_rnn_layers,
            n_epochs=n_epochs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            optimizer_kwargs={"lr": lr},
            batch_size=batch_size,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        blockrnn_GRUmodel = BlockRNNModel(
            model="GRU",
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_rnn_layers=n_rnn_layers,
            n_epochs=n_epochs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            optimizer_kwargs={"lr": lr},
            batch_size=batch_size,
            likelihood=GaussianLikelihood(),
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        nbeats_model = NBEATSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            generic_architecture=True,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            dropout = trial.suggest_float("dropout", 0.1, 0.5),
            n_epochs=n_epochs,
            batch_size=batch_size,
            optimizer_kwargs={"lr": lr},
            likelihood=PositiveGaussianLikelihood(),  # Use the modified likelihood here
            pl_trainer_kwargs=pl_trainer_kwargs, 
        )
        nhits_model = NHiTSModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_widths=layer_widths,
            n_epochs=n_epochs,
            batch_size=batch_size,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        tcn_model = TCNModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            num_layers=num_layers,
            dropout=dropout,
            batch_size=batch_size,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        transformer_model = TransformerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            dropout=dropout,
            batch_size=batch_size,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        dlinear_model = DLinearModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        nlinear_model = NLinearModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            batch_size=batch_size,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        tide_model = TiDEModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            n_epochs=n_epochs,
            hidden_size=trial.suggest_int("hidden_size", 16, 50, log=True),
            dropout=dropout,
            batch_size=batch_size,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )
        tsmixer_model = TSMixerModel(
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            use_reversible_instance_norm=True,
            n_epochs=n_epochs,
            pl_trainer_kwargs=pl_trainer_kwargs,
        )

        model = None
        if model_name == "rnn-model":
            model = rnn_model
        elif model_name == "lstm-model":
            model = lstm_model
        elif model_name == "gru-model":
            model = gru_model
        elif model_name == "tft-model":
            model = tft_model
        elif model_name == "blockrnn-RNNmodel":
            model = blockrnn_RNNmodel
        elif model_name == "blockrnn-LSTMmodel":
            model = blockrnn_LSTMmodel
        elif model_name == "blockrnn-GRUmodel":
            model = blockrnn_GRUmodel
        elif model_name == "nbeats-model":
            model = nbeats_model
        elif model_name == "nhits-model":
            model = nhits_model
        elif model_name == "tcn-model":
            model = tcn_model
        elif model_name == "transformer-model":
            model = transformer_model
        elif model_name == "dlinear-model":
            model = dlinear_model
        elif model_name == "nlinear-model":
            model = nlinear_model
        elif model_name == "tide-model":
            model = tide_model
        elif model_name == "tsmixer-model":
            model = tsmixer_model
        else:
            raise Exception("Model not valid")

        # Fit the model
        # model.fit(train_min)
        if (
            model_name == "rnn-model"
            or model_name == "lstm-model"
            or model_name == "gru-model"
            or model_name == "blockrnn-RNNmodel"
            or model_name == "blockrnn-LSTMmodel"
            or model_name == "blockrnn-GRUmodel"
            or model_name == "nhits-model"
            or model_name == "tcn-model"
            or model_name == "transformer-model"
            or model_name == "dlinear-model"
            or model_name == "nlinear-model"
            or model_name == "tide-model"
            or model_name == "tsmixer-model"
        ):
            model.fit(train_min, dataloader_kwargs={"num_workers": 8})
        else:
            model.fit(train_min)

        # Make predictions and evaluate
        forecast = model.predict(len(val_min))
        error = root_mean_squared_error(val_min.values(), forecast.values())
        if error < best_score:
            best_score = error
            best_model = model
            save_path = f"/mnt/tuned-models-volume/global/1512/{round(error, 6)}_best_model_trial_{trial.number}-{model_name}.pth"
            best_model = model
            best_model.save(save_path)
        return error

    # Create Optuna study and optimize
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)
    except Exception as err:
        print(f"\nSTUDY FAILED: {err=}, {type(err)=}\n")

models = ["nbeats-model"]

for model in models:
    print(f"Bigginning training on: {model}")
    try:
        tune_model(model, 75)
        print(f"Done with: {model}")
    except Exception as err:
        print(f"Error: {err=}, {type(err)=}")

print("Done domingue")
