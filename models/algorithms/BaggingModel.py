from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from typing import Any, Tuple
from models.BaseModelABC import BaseModelABC
from models.Evaluator import Evaluator
from decorators.ArgsChecker import ArgsChecker

class BaggingModel(BaseModelABC):
    def __init__(self):
        self.model = BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=42)

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
        return pd.Series(predictions)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[float, float, float, float]:
        return Evaluator.evaluate_model(self, X_test, y_test)
