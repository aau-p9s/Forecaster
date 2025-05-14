from uuid import UUID


class Setting:
    def __init__(self, id:UUID, service_id:UUID, scale_up:int, scale_down:int, min_replicas:int, max_replicas:int, scale_period:int, train_interval:int, hyperparameters:dict, optuna_config:dict) -> None:
        self.id = id
        self.service_id = service_id
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_period = scale_period
        self.train_interval = train_interval
        self.hyperparameters = hyperparameters
        self.optuna_config = optuna_config

