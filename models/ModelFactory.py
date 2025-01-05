# opti-ml-py\models\ModelFactory.py
from models.LightGBMModel import LightGBMModel
from models.RandomForestModel import RandomForestModel
from models.XGBoostModel import XGBoostModel
from models.CatBoostModel import CatBoostModel
from models.AdaBoostModel import AdaBoostModel
from models.SVMModel import SVMModel
from models.KNeighborsModel import KNeighborsModel
from models.LogisticRegressionModel import LogisticRegressionModel
from decorators.ArgsChecker import ArgsChecker
from typing import List
from models.BaseModelABC import BaseModelABC


class ModelFactory:
    @ArgsChecker((None, str), BaseModelABC)
    def create_model(self, model_type: str) -> BaseModelABC:
        if model_type == "lightgbm":
            return LightGBMModel()
        elif model_type == "rand_frst":
            return RandomForestModel()
        elif model_type == "xgboost":
            return XGBoostModel()
        elif model_type == "catboost":
            return CatBoostModel()
        elif model_type == "adaboost":
            return AdaBoostModel()
        elif model_type == "svm":
            return SVMModel()
        elif model_type == "knn":
            return KNeighborsModel()
        elif model_type == "logc_regr":
            return LogisticRegressionModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @ArgsChecker((None, list), list)
    def create_models(self, model_types: list[str]) -> list[BaseModelABC]:
        return [self.create_model(model_type) for model_type in model_types]
