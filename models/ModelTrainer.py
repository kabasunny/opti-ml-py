# opti-ml-py\models\ModelTrainer.py
import pandas as pd
from typing import List
from decorators.ArgsChecker import ArgsChecker
from models.BaseModelABC import BaseModelABC


class ModelTrainer:
    @staticmethod
    @ArgsChecker(
        (List[BaseModelABC], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series),
        List[BaseModelABC],
    )
    def train(
        models: List[BaseModelABC],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> List[BaseModelABC]:
        for model in models:
            model.train(X_train, y_train, X_test, y_test)
        return models

    @staticmethod
    @ArgsChecker((List[BaseModelABC], pd.DataFrame, pd.Series), None)
    def evaluate(models: List[BaseModelABC], X_test: pd.DataFrame, y_test: pd.Series):
        for model in models:
            model.evaluate(X_test, y_test)
