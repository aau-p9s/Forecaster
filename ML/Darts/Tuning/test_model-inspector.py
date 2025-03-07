from model_inspector import inspect_model
from darts.models import RNNModel, ExponentialSmoothing, BATS

# Load the trained model
model = ExponentialSmoothing.load("/mnt/tuned-models-volume/statistical/best_hp_models/0.414253_best_model_trial_66-exponentialsmoothing_model.pth")

# Inspect the model
inspect_model(model)
