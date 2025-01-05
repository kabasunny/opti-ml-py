# opti-ml-py\models\CatBoostModel.py
from catboost import CatBoostClassifier
import pandas as pd
from typing import Any
from models.BaseModelABC import BaseModelABC
from models.Evaluator import Evaluator
from decorators.ArgsChecker import ArgsChecker


class CatBoostModel(BaseModelABC):
    def __init__(self):
        self.model = CatBoostClassifier(
            iterations=1000, learning_rate=0.01, depth=6, verbose=0
        )

    @ArgsChecker((None, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series), None)
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        self.model.fit(X_train, y_train)
        self.evaluate(X_test, y_test)

    @ArgsChecker((None, pd.DataFrame), pd.Series)
    def predict(self, X_test: pd.DataFrame) -> Any:
        return self.model.predict(X_test)

    @ArgsChecker((None, pd.DataFrame, pd.Series), None)
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        Evaluator.evaluate_model(self.model, X_test, y_test)
