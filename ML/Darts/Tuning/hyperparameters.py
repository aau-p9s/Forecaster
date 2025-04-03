from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode
import optuna
import inspect
from darts.dataprocessing.transformers import Scaler

class HyperParameterConfig:

    accepts_n_epochs = [] # Models which accept number of epochs, should maybe be set at fixed number
    _P = None
    def __init__(self, trial, model_class, series):
        """
        Initialize hyperparameter configuration for a specific model class.
        
        Parameters:
        - trial: optuna.trial.Trial instance.
        - model_class: The model class for which to generate hyperparameters.
        """
        self.model_class = model_class
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
        
        def encode_time(idx):
            return (idx.year - 2025) + (idx.dayofyear / 365) + (idx.hour / 24) + (idx.minute / 1440)
        self.ENCODERS = {
            "month": {
                'cyclic': {'future': ['month', 'minute']},  # Add minute-level cyclic encoding
                'datetime_attribute': {'future': ['hour', 'dayofweek', 'dayofyear']},  # Add dayofyear
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [encode_time]},  # Use improved time encoding
                'transformer': Scaler(),
                'tz': 'CET'
            },
            "none": None
        }

        self.SEASONALITIES = {
            "daily": dict({
                'name': "daily",  # (name of the seasonality component),
                'seasonal_periods': 1440, # (nr of steps composing a season),
                'mode': "additive"  # ('additive' or 'multiplicative')
            }),
            "default": None
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
            self._P = P

        if "p" in self.valid_params:
            # Ensure `p` is not equal to `s`
            parameters["p"] = trial.suggest_int("p", 0, 12)
            
            while self._P is not None and self._P * s == parameters.get("p"):
                P = trial.suggest_int("P", 0, 5)
            
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

            parameters["trend"] = trial.suggest_categorical("trend", ["n", "c", "t", "ct"])  # Flexible trend selection
            if parameters.get("d") + D > 0 and parameters.get("trend") == "c":
                raise optuna.TrialPruned()  # Skip invalid trials

        if "hidden_fc_sizes" in self.valid_params:
            parameters["hidden_fc_sizes"] = trial.suggest_int("hidden_fc_size", 8, 256)

        if "symmetric" in self.valid_params:
            parameters["symmetric"] = trial.suggest_categorical("symmetric", [False, True])
        
        if "cal_length" in self.valid_params:
            parameters["cal_length"] = trial.suggest_categorical("cal_length", None)
        
        if "cal_stride" in self.valid_params:
            parameters["cal_stride"] = trial.suggest_int("cal_stride", 1, 10)
        
        if "cal_num_samples" in self.valid_params:
            parameters["cal_num_samples"] = trial.suggest_int("cal_num_samples", 1, 10)

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

        if "model" in self.valid_params:
            parameters["model"] = trial.suggest_categorical("model", ["LSTM", "GRU", "RNN"])

        if "num_layers_out_fc" in self.valid_params:
            # Suggest the number of layers (1 to 3 layers)
            num_layers = trial.suggest_int("num_layers_out_fc", 1, 3)  # Number of layers between 1 and 3
            
            # Suggest sizes for each layer (32 to 512 neurons)
            layers_sizes = [
                trial.suggest_int(f"layer_size_{i}", 32, 512) for i in range(num_layers)
            ]
            
            # Set the suggested list of layer sizes to the parameters
            parameters["num_layers_out_fc"] = layers_sizes

        if "nr_epochs_val_period" in self.valid_params:
            parameters["nr_epochs_val_period"] = trial.suggest_int("nr_epochs_val_period", 1, 20)

        if "hidden_dim" in self.valid_params:
            parameters["hidden_dim"] = trial.suggest_int("hidden_dim", 10, 100)

        if "dropout" in self.valid_params:
            parameters["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

        if "n_rnn_layers" in self.valid_params:
            parameters["n_rnn_layers"] = trial.suggest_int("n_rnn_layers", 1, 3)

        if "batch_size" in self.valid_params:
            parameters["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])

        if "n_epochs" in self.valid_params:
            parameters["n_epochs"] = trial.suggest_int("n_epochs", 10, 100)

        if "training_length" in self.valid_params:
            parameters["training_length"] = trial.suggest_int("training_length", 1, len(self.series))

        if "activation" in self.valid_params:
            if "TSMixer" in self.model_class.__name__:
                parameters["activation"] = trial.suggest_categorical("activation", ['ReLU', 'RReLU', 'PReLU', 'ELU', 'Softplus', 'Tanh', 'SELU', 'LeakyReLU', 'Sigmoid', 'GELU'])
            else:
                parameters["activation"] = trial.suggest_categorical("activation", ['GLU', 'Bilinear', 'ReGLU', 'GEGLU', 'SwiGLU', 'ReLU', 'GELU', 'relu', 'gelu'])

        if "lr" in self.valid_params:
            parameters["lr"] = trial.suggest_float("lr", 5e-5, 1e-3, log=True)

        if "num_blocks" in self.valid_params:
            parameters["num_blocks"] = trial.suggest_int("num_blocks", 1, 3)

        if "num_layers" in self.valid_params:
            parameters["num_layers"] = trial.suggest_int("num_layers", 1, 3)

        if "generic_architecture" in self.valid_params:
            parameters["generic_architecture"] = trial.suggest_categorical("generic_architecture", [True, False])
            if parameters.get("generic_architecture", False):
                if "expansion_coefficient_dim" in self.valid_params:
                    parameters["expansion_coefficient_dim"] = trial.suggest_int("expansion_coefficient_dim", 1, 50)
            else:
                if "trend_polynomial_degree" in self.valid_params:
                    parameters["trend_polynomial_degree"] = trial.suggest_int("trend_polynomial_degree", 1, 50)

        if "num_stacks" in self.valid_params:
            parameters["num_stacks"] = trial.suggest_int("num_stacks", 1, 5) if trial.suggest_categorical("generic_architecture", [True, False]) else None

        if "layers_widths" in self.valid_params:
            parameters["layers_widths"] = trial.suggest_int("layer_widths", 32, 256)

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
            if "ExponentialSmoothing" in self.model_class.__name__:
                trend_str = trial.suggest_categorical("trend", list(self.MODEL_MODES.keys()))
                parameters["trend"] = self.MODEL_MODES[trend_str]
            elif "FFT" in self.model_class.__name__:
                parameters["trend"] = trial.suggest_categorical("trend", ["poly", "exp"])
            else:
                parameters["trend"] = trial.suggest_categorical('trend', ["n", "c", "t", "ct"])
        
        if "trend_poly_degree" in self.valid_params and parameters["trend"] == "poly":
            parameters["trend_poly_degree"] = trial.suggest_int("trend_poly_degree", 1, 5)

        if "model_mode" in self.valid_params:
            model_mode_str = trial.suggest_categorical("model_mode", self.MODEL_MODES.keys())
            parameters["mod"] = self.MODEL_MODES[model_mode_str]

        if "seasonal" in self.valid_params:
            seasonal_str = trial.suggest_categorical("seasonal", list(self.SEASONALITY_MODES.keys()))
            parameters["seasonal"] = self.SEASONALITY_MODES[seasonal_str]

        if "add_encoders" in self.valid_params: # Possibly add more encodings
            encoder_str = trial.suggest_categorical("add_encoders", list(self.ENCODERS.keys()))
            parameters["add_encoders"] = self.ENCODERS[encoder_str]
        
        if "add_seasonalities" in self.valid_params: 
            seasonalities_str = trial.suggest_categorical("add_seasonalities", list(self.SEASONALITIES.keys()))
            parameters["add_seasonalities"] = self.SEASONALITIES[seasonalities_str]

        if "country_holidays" in self.valid_params:
            parameters["country_holidays"] = trial.suggest_categorical("country_holidays", ["DK", None])

        if "cap" in self.valid_params:
            parameters["cap"] = trial.suggest_float("cap", 100.0, 10000.0, log=True)

        if "floor" in self.valid_params:
            parameters["floor"] = trial.suggest_float("floor", 0.0, parameters["cap"] * 0.5)

        if "season_mode" in self.valid_params:
            season_str = trial.suggest_categorical("season_mode", list(self.SEASONALITY_MODES.keys()))
            parameters["season_mode"] = self.SEASONALITY_MODES[season_str]

        if "damped" in self.valid_params:
            if parameters.get("trend") == ModelMode.NONE:
                parameters["damped"] = False
            else:
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
        
        if "seasonality_period" in self.valid_params:
            parameters["seasonality_period"] = trial.categorical("seasonality_period", None) # Is inferred from fit data
        
        if "seasonal_periods" in self.valid_params:
            parameters["seasonal_periods"] = trial.categorical("seasonal_periods", ["freq"], 7, 12, 24)

        if "use_arma_errors" in self.valid_params:
            parameters["use_arma_errors"] = trial.categorical("use_arma_errors", [True, False, None])
        
        if "theta" in self.valid_params:
            parameters["theta"] = trial.suggest_int("theta", 1, 3)

        if "num_encoder_layers" in self.valid_params:
            parameters["num_encoder_layers"] = trial.suggest_int("num_encoder_layers", 1, 12)

        if "num_decoder_layers" in self.valid_params:
            parameters["num_decoder_layers"] = trial.suggest_int("num_decoder_layers", 1, 12)

        if "decoder_output_dim" in self.valid_params:
            parameters["decoder_output_dim"] = trial.suggest_int("decoder_output_dim", 1, 12)

        if "ff_size" in self.valid_params:
            parameters["ff_size"] = trial.suggest_int("ff_size", 1, 12)

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
        
        if "normalization" in self.valid_params:
            parameters["normalization"] = trial.suggest_categorical("normalization", [True, False])

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
            parameters["num_attention_heads"] = trial.suggest_int("num_attention_heads", 3, 6)

        if "const_init" in self.valid_params:
            parameters["const_init"] = trial.suggest_categorical("const_init", [True, False])

        if "shared_weights" in self.valid_params:
            parameters["shared_weights"] = trial.suggest_categorical("shared_weights", [True, False])

        if "num_attention_heads" in self.valid_params:
            parameters["num_attention_heads"] = trial.suggest_int("num_attention_heads", 3, 6)

        if "full_attention" in self.valid_params:
            parameters["full_attention"] = trial.suggest_categorical("full_attention", [True, False])

        if "feed_forward" in self.valid_params:
            parameters["feed_forward"] = trial.suggest_categorical("feed_forward", ["GLU", "Bilinear", "ReGLU", "GEGLU", "SwiGLU", "ReLU", "GELU", "GatedResidualNetwork"])

        if "hidden_continuous_size" in self.valid_params:
            parameters["hidden_continuous_size"] = trial.suggest_int("hidden_continuous_size", 8, 256)
            
        if "dim_feedforward" in self.valid_params:
            parameters["dim_feedforward"] = trial.suggest_int("dim_feedforward", 10, 518)
            
        if "season_length" in self.valid_params:
            parameters["season_length"] = trial.suggest_int("season_length", 12, 2880)
        
        if "nr_freqs_to_keep" in self.valid_params:
            parameters["nr_freqs_to_keep"] = trial.suggest_int("nr_freqs_to_keep", 100, 2880)

        if "model" in self.valid_params and self.model_class.__name__ not in ["BlockRNNModel", "RNNModel"]:
            parameters["model"] = "Z"
        
        if "add_relative_index" in self.valid_params:
            parameters["add_relative_index"] = trial.suggest_categorical("add_relative_index", [True])
        
        if "trend_mode" in self.valid_params:
            trend_mode_str = trial.suggest_categorical("trend_mode", self.TREND_MODES.keys())
            parameters["trend_mode"] = self.TREND_MODES[trend_mode_str]
            
        if "norm_type" in self.valid_params:
            parameters["norm_type"] = trial.suggest_categorical("trend_mode", ["LayerNorm", "RMSNorm", "LayerNormNoBias", None])

        if "multi_models" in self.valid_params:
            parameters["multi_models"] = trial.suggest_categorical("multi_models", [True, False])
        
        if "use_static_covariates" in self.valid_params:
            parameters["use_static_covariates"] = trial.suggest_categorical("use_static_covariates", [True, False])

        if "likelihood" in self.valid_params:
            parameters["likelihood"] = trial.suggest_categorical("likelihood", ["quantile", "poisson", None])
            if parameters["likelihood"] == "quantile":
                if "quantiles" in self.valid_params:
                    parameters["quantiles"] = trial.suggest_categorical("quantiles", [[0.1, 0.25, 0.5, 0.75, 0.9], [0.5], [0.01, 0.5, 0.99]])

        
        if "n_estimators" in self.valid_params:
            parameters["n_estimators"] = trial.suggest_int("n_estimators", 1, 200)

        if "max_depth" in self.valid_params:
            parameters["max_depth"] = trial.suggest_categorical("max_depth", [1, 50, 100, 200, None])

        for param, value in parameters.items():
            setattr(self, param, value)

    def suggest_seasonal_order(self, trial):
        """
        Suggest seasonal order for models requiring it (e.g., ARIMA).
        """
        P = trial.suggest_int("P", 0, 5)
        D = trial.suggest_int("D", 0, 2)
        Q = trial.suggest_int("Q", 0, 5)
        s = trial.suggest_categorical("s", [3, 4, 6, 7, 9, 12])
        return (P, D, Q, s)
