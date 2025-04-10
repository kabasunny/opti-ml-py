# opti-ml-py\models\XGBoostModel.py
import xgboost as xgb
import pandas as pd
from typing import Any, Tuple
from models.BaseModelABC import BaseModelABC
from models.Evaluator import Evaluator
from decorators.ArgsChecker import ArgsChecker


class XGBoostModel(BaseModelABC):
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42
        )

    # @ArgsChecker(
    #     (None, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series),
    #     Tuple["BaseModelABC", Tuple[float, float, float, float]],
    # )
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple["BaseModelABC", Tuple[float, float, float, float]]:
        self.model.fit(X_train, y_train)
        result = self.evaluate(X_test, y_test)
        return self, result

    @ArgsChecker((None, pd.DataFrame), pd.Series)
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X_test)
        binary_predictions = (predictions >= 0.5).astype(int)
        return pd.Series(binary_predictions)

    # @ArgsChecker((None, pd.DataFrame, pd.Series), Tuple[float, float, float, float])
    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[float, float, float, float]:
        result = Evaluator.evaluate_model(self.model, X_test, y_test)
        return result
