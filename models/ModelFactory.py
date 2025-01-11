from models.LightGBMModel import LightGBMModel
from models.RandomForestModel import RandomForestModel
from models.XGBoostModel import XGBoostModel
from models.CatBoostModel import CatBoostModel
from models.AdaBoostModel import AdaBoostModel
from models.SVMModel import SVMModel
from models.KNeighborsModel import KNeighborsModel
from models.LogisticRegressionModel import LogisticRegressionModel
from decorators.ArgsChecker import ArgsChecker
from models.BaseModelABC import BaseModelABC


class ModelFactory:
    @staticmethod
    @ArgsChecker((None, str), BaseModelABC)
    def create_model(model_type: str) -> BaseModelABC:
        if model_type == "LightGBM":
            return LightGBMModel()
        elif model_type == "RandomForest":
            return RandomForestModel()
        elif model_type == "XGBoost":
            return XGBoostModel()
        elif model_type == "CatBoost":
            return CatBoostModel()
        elif model_type == "AdaBoost":
            return AdaBoostModel()
        elif model_type == "SVM":
            return SVMModel()
        elif model_type == "KNeighbors":
            return KNeighborsModel()
        elif model_type == "LogisticRegression":
            return LogisticRegressionModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    @ArgsChecker((None, list[BaseModelABC]), list[BaseModelABC])
    def create_models(model_types: list[str]) -> list[BaseModelABC]:
        # print(model_types)
        return [ModelFactory.create_model(model_type) for model_type in model_types]
