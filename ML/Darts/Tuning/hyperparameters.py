from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
import optuna
import inspect

class HyperParameterConfig:

    accepts_n_epochs = [] # Models which accept number of epochs, should maybe be set at fixed number

    def __init__(self, trial, model_class, series):
        """
        Initialize hyperparameter configuration for a specific model class.
        
        Parameters:
        - trial: optuna.trial.Trial instance.
        - model_class: The model class for which to generate hyperparameters.
        """
        
        self.MODEL_MODES = {
            "additive": ModelMode.ADDITIVE,
            "multiplicative": ModelMode.MULTIPLICATIVE,
            "none": ModelMode.NONE
        }
        self.SEASONALITY_MODES = {
            "additive": SeasonalityMode.ADDITIVE,
            "multiplicative": SeasonalityMode.MULTIPLICATIVE,
            "none": SeasonalityMode.NONE
        }
        self.TREND_MODES = {
            "linear": TrendMode.LINEAR,
            "exponential": TrendMode.EXPONENTIAL
        }
        self.series = series
        self.valid_params = self.get_valid_params(model_class)
        self.suggest_parameters(trial)

    def get_valid_params(self, model_class):
        """
        Get valid parameters for the specified model class.
        """
        # Inspect the model's __init__ method signature
        init_signature = inspect.signature(model_class.__init__)
        return list(init_signature.parameters.keys())[1:]  # Exclude 'self'

    def suggest_parameters(self, trial):
        """
        Suggest hyperparameters based on the valid parameters of the model class.
        """
        
        parameters = {}

        # Check if seasonal_order is in valid parameters
        if "seasonal_order" in self.valid_params:
            parameters["seasonal_order"] = self.suggest_seasonal_order(trial)

            # Extract seasonal order components, or set default values if not provided
            D, P, Q, s = parameters.get("seasonal_order", (0, 0, 0, 0))

        # Suggest non-seasonal ARIMA parameters
        if "p" in self.valid_params:
            parameters["p"] = trial.suggest_int("p", 0, 12)
        if "d" in self.valid_params:
            parameters["d"] = trial.suggest_int("d", 0, 2)  # Limit differencing to avoid over-differencing
        if "q" in self.valid_params:
            parameters["q"] = trial.suggest_int("q", 0, 12)

        # Ensure P, Q are distinct from p, q
        if "seasonal_order" in self.valid_params:
            parameters["P"] = P
            while parameters["P"] == parameters["p"]:  # Avoid conflicts in AR terms
                parameters["P"] = trial.suggest_int("P", 0, 5)

            parameters["Q"] = Q
            while parameters["Q"] == parameters["q"]:  # Avoid conflicts in MA terms
                parameters["Q"] = trial.suggest_int("Q", 0, 5)

            parameters["s"] = s  # Ensure s > 1 for seasonality

            # Compute total differencing
            total_differencing = parameters["d"] + D

            # Set trend based on total differencing
            if total_differencing >= 2:
                parameters["trend"] = None  # Differencing eliminates lower-order trends
            elif total_differencing == 1:
                parameters["trend"] = "t"  # Linear trend allowed
            else:
                parameters["trend"] = trial.suggest_categorical("trend", ["n", "c", "t", "ct"])  # Flexible trend selection

        if "hidden_fc_sizes" in self.valid_params:
            parameters["hidden_fc_sizes"] = trial.suggest_int("hidden_fc_size", 8, 256)

        if "input_chunk_length" in self.valid_params:
            parameters["input_chunk_length"] = trial.suggest_int("input_chunk_length", 50, len(self.series) // 2)

        if "output_chunk_length" in self.valid_params:
            parameters["output_chunk_length"] = trial.suggest_int("output_chunk_length", 1, 24)

        if "output_chunk_shift" in self.valid_params and "input_chunk_length" in self.valid_params:
            # Ensure output_chunk_shift is 0 if auto-regression is used (n > output_chunk_length)
            if parameters["input_chunk_length"] > parameters["output_chunk_length"]:
                parameters["output_chunk_shift"] = 0  # No shifting allowed in auto-regression
            else:
                parameters["output_chunk_shift"] = trial.suggest_int("output_chunk_shift", 0, 10)



        if "hidden_dim" in self.valid_params:
            parameters["hidden_dim"] = trial.suggest_int("hidden_dim", 10, 100)

        if "dropout" in self.valid_params:
            parameters["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

        if "n_rnn_layers" in self.valid_params:
            parameters["n_rnn_layers"] = trial.suggest_int("n_rnn_layers", 1, 3)

        if "batch_size" in self.valid_params:
            parameters["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])

        if "n_epochs" in self.valid_params:
            parameters["n_epochs"] = trial.suggest_int("n_epochs", 10, 100)

        if "training_length" in self.valid_params:
            parameters["training_length"] = trial.suggest_int("training_length", 1, len(self.series))

        if "lr" in self.valid_params:
            parameters["lr"] = trial.suggest_float("lr", 5e-5, 1e-3, log=True)

        if "num_blocks" in self.valid_params:
            parameters["num_blocks"] = trial.suggest_int("num_blocks", 1, 3)

        if "num_layers" in self.valid_params:
            parameters["num_layers"] = trial.suggest_int("num_layers", 1, 3)

        if "generic_architecture" in self.valid_params:
            parameters["generic_architecture"] = trial.suggest_categorical("generic_architecture", [True, False])

        if "num_stacks" in self.valid_params:
            parameters["num_stacks"] = trial.suggest_int("num_stacks", 1, 5) if trial.suggest_categorical("generic_architecture", [True, False]) else None

        if "layers_widths" in self.valid_params:
            parameters["layers_widths"] = trial.suggest_int("layer_widths", 32, 256)

        while all(param in self.valid_params for param in ["lags", "lags_past_covariates", "lags_future_covariates"]) and \
            not any(parameters.get(param) is not None for param in ["lags", "lags_past_covariates", "lags_future_covariates"]):
    
            parameters = {}

            if "lags" in self.valid_params:
                value = trial.suggest_categorical("lags", [None, -1, -3, -5, -10])
                parameters["lags"] = None if value is None else [value]

            if "lags_past_covariates" in self.valid_params:
                value = trial.suggest_categorical("lags_past_covariates", [None, -1, -3, -5, -10])
                parameters["lags_past_covariates"] = [value] if isinstance(value, int) else value

            if "lags_future_covariates" in self.valid_params:
                value = trial.suggest_categorical("lags_future_covariates", [None, 1, 3, 5, 10])
                parameters["lags_future_covariates"] = None if value is None else [value]



        if "start_p" in self.valid_params:
            parameters["start_p"] = trial.suggest_int("start_p", 0, 8)

        if "max_p" in self.valid_params:
            parameters["max_p"] = trial.suggest_int("max_p", trial.suggest_int("start_p", 0, 8), 12)

        if "start_q" in self.valid_params:
            parameters["start_q"] = trial.suggest_int("start_q", 0, 1)

        if "trend" in self.valid_params:
            parameters["trend"] = trial.suggest_categorical('trend', ["n", "c", "t", "ct"])

        if "model_mode" in self.valid_params:
            parameters["model_mode"] = trial.suggest_categorical("model_mode", self.MODEL_MODES.keys())

        if "seasonal" in self.valid_params:
            parameters["seasonal"] = trial.suggest_categorical("seasonal", self.SEASONALITY_MODES.keys())

        if "damped" in self.valid_params:
            parameters["damped"] = trial.suggest_categorical("damped", [True, False]) if self.MODEL_MODES[trial.suggest_categorical("model_mode", self.MODEL_MODES.keys())] != ModelMode.NONE else False

        if "use_damped_trend" in self.valid_params:
            parameters["use_damped_trend"] = trial.suggest_categorical("use_damped_trend", [True, False, None])

        if "use_trend" in self.valid_params:
            parameters["use_trend"] = trial.suggest_categorical("use_trend", [True, False, None])

        if "box_cox_bounds_lower" in self.valid_params:
            parameters["box_cox_bounds_lower"] = trial.suggest_int("box_cox_bounds_lower", -5, 1)

        if "box_cox_bounds_upper" in self.valid_params:
            parameters["box_cox_bounds_upper"] = trial.suggest_int("box_cox_bounds_upper", trial.suggest_int("box_cox_bounds_lower", -5, 1), 5)

        if "use_box_cox" in self.valid_params:
            parameters["use_box_cox"] = trial.suggest_categorical("use_box_cox", [True, False])

        if "theta" in self.valid_params:
            parameters["theta"] = trial.suggest_int("theta", 1, 3)

        if "num_encoder_layers" in self.valid_params:
            parameters["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 1, 12)

        if "num_decoder_layers" in self.valid_params:
            parameters["num_decoder_layers"] = trial.suggest_int("num_decoder_layers", 1, 12)

        if "decoder_output_dim" in self.valid_params:
            parameters["decoder_output_dim"] = trial.suggest_int("decoder_output_dim", 1, 12)

        if "temporal_width_past" in self.valid_params:
            parameters["temporal_width_past"] = trial.suggest_int("temporal_width_past", 1, 168)

        if "temporal_width_future" in self.valid_params:
            parameters["temporal_width_future"] = trial.suggest_int("temporal_width_future", 1, 168)

        if "temporal_hidden_size_past" in self.valid_params:
            parameters["temporal_hidden_size_past"] = trial.suggest_int("temporal_hidden_size_past", 1, 168)

        if "temporal_hidden_size_future" in self.valid_params:
            parameters["temporal_hidden_size_future"] = trial.suggest_int("temporal_hidden_size_future", 1, 168)

        if "temporal_decoder_hidden" in self.valid_params:
            parameters["temporal_decoder_hidden"] = trial.suggest_int("temporal_decoder_hidden", 1, 12)

        if "use_layer_norm" in self.valid_params:
            parameters["use_layer_norm"] = trial.suggest_categorical("use_layer_norm", [True, False])

        if "d_model" in self.valid_params:
            parameters["d_model"] = trial.suggest_int("d_model", 32, 256)

        if "nhead" in self.valid_params:
            parameters["nhead"] = trial.suggest_int("nhead", 2, 8)

        if "dim_x" in self.valid_params:
            parameters["dim_x"] = trial.suggest_int("dim_x", 5, 20)

        if "num_filters" in self.valid_params:
            parameters["num_filters"] = trial.suggest_int("num_filters", 1, 100)

        if "weight_norm" in self.valid_params:
            parameters["weight_norm"] = trial.suggest_categorical("weight_norm", [True, False, None])

        if "dilation_base" in self.valid_params:
            parameters["dilation_base"] = trial.suggest_int("dilation_base", 0, 10)

        if "version" in self.valid_params:
            parameters["version"] = trial.suggest_categorical("version", ["optimized", "classic", "sba", "tsb"])

        if "alpha_d" in self.valid_params:
            parameters["alpha_d"] = trial.suggest_float("alpha_d", 0.05, 0.3)

        if "alpha_p" in self.valid_params:
            parameters["alpha_p"] = trial.suggest_float("alpha_p", 0.05, 0.3)

        if "kernel_size" in self.valid_params:
            parameters["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])

        if "hidden_size" in self.valid_params:
            parameters["hidden_size"] = trial.suggest_int("hidden_size", 16, 256)

        if "num_attention_heads" in self.valid_params:
            parameters["num_attention_heads"] = trial.suggest_int("num_attention_heads", 1, 8)

        if "feed_forward" in self.valid_params:
            parameters["feed_forward"] = trial.suggest_categorical("feed_forward", ["GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU", "ReLU", "GELU"])

        if "hidden_continuous_size" in self.valid_params:
            parameters["hidden_continuous_size"] = trial.suggest_int("hidden_continuous_size", 8, 256)
            
        if "dim_feedforward" in self.valid_params:
            parameters["dim_feedforward"] = trial.suggest_int("dim_feedforward", 10, 518)
            
        if "season_length" in self.valid_params:
            parameters["season_length"] = trial.suggest_int("season_length", 12, 2880)

        if "model" in self.valid_params:
            parameters["model"] = "Z"
            
        if "trend_mode" in self.valid_params:
            parameters["trend_mode"] = trial.suggest_categorical("trend_mode", self.TREND_MODES.keys())
            
        if "norm_type" in self.valid_params:
            parameters["norm_type"] = trial.suggest_categorical("trend_mode", ["LayerNorm", "RMSNorm", "LayerNormNoBias", None])

        for param, value in parameters.items():
            setattr(self, param, value)

    def suggest_seasonal_order(self, trial):
        """
        Suggest seasonal order for models requiring it (e.g., ARIMA).
        """
        P = trial.suggest_int("P", 0, 5)
        D = trial.suggest_int("D", 0, 2)
        Q = trial.suggest_int("Q", 0, 5)
        s = trial.suggest_int("s", 2, 12)
        return (P, D, Q, s)
