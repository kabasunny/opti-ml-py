from typing import List, Tuple
import pandas as pd
from decorators.ArgsChecker import ArgsChecker
from models.BaseModelABC import BaseModelABC


class ModelTrainer:
    @staticmethod
    def train(
        models: List[BaseModelABC],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[List[BaseModelABC], pd.DataFrame]:
        results = []
        trained_models = []
        model_names = []  # モデルの名前を格納するリスト
        for model in models:
            trained_model, result = model.train(X_train, y_train, X_test, y_test)
            trained_models.append(trained_model)
            results.append(result)
            model_names.append(
                type(model).__name__.replace("Model", "")
            )  # "Model" を除去
        results_df = pd.DataFrame(
            results,
            columns=["Accuracy", "Precision", "Recall", "F1"],
            index=model_names,
        )
        return trained_models, results_df

    @staticmethod
    def evaluate(
        models: List[BaseModelABC], X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        results = []
        model_names = []  # モデルの名前を格納するリスト
        for model in models:
            result = model.evaluate(X_test, y_test)
            results.append(result)
            model_names.append(
                type(model).__name__.replace("Model", "")
            )  # "Model" を除去
        results_df = pd.DataFrame(
            results,
            columns=["Accuracy", "Precision", "Recall", "F1"],
            index=model_names,
        )
        return results_df
