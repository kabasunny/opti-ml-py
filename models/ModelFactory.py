from models.algorithms.AdaBoostModel import AdaBoostModel
from models.algorithms.CatBoostModel import CatBoostModel
from models.algorithms.DecisionTreeModel import DecisionTreeModel
from models.algorithms.GradientBoostingModel import GradientBoostingModel
from models.algorithms.KNeighborsModel import KNeighborsModel
from models.algorithms.LightGBMModel import LightGBMModel
from models.algorithms.LogisticRegressionModel import LogisticRegressionModel
from models.algorithms.NaiveBayesModel import NaiveBayesModel
from models.algorithms.RandomForestModel import RandomForestModel
from models.algorithms.RidgeRegressionModel import RidgeRegressionModel
from models.algorithms.SVMModel import SVMModel
from models.algorithms.XGBoostModel import XGBoostModel
from models.algorithms.ExtraTreesModel import ExtraTreesModel
from models.algorithms.BaggingModel import BaggingModel
from models.algorithms.VotingModel import VotingModel
from models.algorithms.StackingModel import StackingModel
from models.algorithms.PassiveAggressiveModel import PassiveAggressiveModel
from models.algorithms.PerceptronModel import PerceptronModel
from models.algorithms.SGDModel import SGDModel
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
        elif model_type == "DecisionTree":
            return DecisionTreeModel()
        elif model_type == "GradientBoosting":
            return GradientBoostingModel()
        elif model_type == "NaiveBayes":
            return NaiveBayesModel()
        elif model_type == "RidgeRegression":
            return RidgeRegressionModel()
        elif model_type == "ExtraTrees":
            return ExtraTreesModel()
        elif model_type == "Bagging":
            return BaggingModel()
        elif model_type == "Voting":
            return VotingModel()
        elif model_type == "Stacking":
            return StackingModel()
        elif model_type == "PassiveAggressive":
            return PassiveAggressiveModel()
        elif model_type == "Perceptron":
            return PerceptronModel()
        elif model_type == "SGD":
            return SGDModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    @ArgsChecker((None, list[BaseModelABC]), list[BaseModelABC])
    def create_models(model_types: list[str]) -> list[BaseModelABC]:
        return [ModelFactory.create_model(model_type) for model_type in model_types]
