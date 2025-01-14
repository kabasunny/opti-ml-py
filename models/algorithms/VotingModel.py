from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from typing import Any, Tuple
from models.BaseModelABC import BaseModelABC
from models.Evaluator import Evaluator
from decorators.ArgsChecker import ArgsChecker

class VotingModel(BaseModelABC):
    def __init__(self):
        self.model = VotingClassifier(estimators=[
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('rf', RandomForestClassifier())
        ], voting='hard')

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
