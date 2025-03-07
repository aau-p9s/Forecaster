import optuna
from hyperparameters import HyperParameterConfig  # Import config from the external file
import darts.models as models
from darts.models.forecasting.forecasting_model import ForecastingModel
import inspect
from pytorch_lightning.callbacks import EarlyStopping
import logging, sys

# Get all forecasting models
models = [
    cls for name, cls in vars(models).items()
    if inspect.isclass(cls) and issubclass(cls, ForecastingModel)
]

# Optuna vars
storage_name = "sqlite:///Tuning/model-tuning.db"
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Select the model class (change index as needed for different models)
model_class = models[35]
model_name = model_class().__class__.__name__

def objective(trial, model_class):
    
    # Initialize the config for hyperparameter suggestions
    config = HyperParameterConfig(trial, model_class)


    try:
        params = {key: getattr(config, key) for key in config.valid_params if hasattr(config, key)}
        model = model_class(**params)
        print(f"Successfully instantiated model: {model.__class__.__name__}")
        # Add training and evaluation here...
        print(trial.user_attrs)
    except Exception as e:
        print(f"Failed to instantiate {model_class.__name__}: {e}")

    del model_class
    return 0

# Run Optuna study
study = optuna.create_study(
    study_name=model_name + "_study",
    direction="minimize",
    pruner=optuna.pruners.PatientPruner(wrapped_pruner=None, min_delta=0.05, patience=1), # This is a pruner which acts as an early stopper
    storage=storage_name, load_if_exists=True
)

study.optimize(lambda trial: objective(trial, model_class), n_trials=50)