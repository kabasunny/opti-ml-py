# opti-ml-py\models\BaseModelABC.py
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from decorators.ArgsChecker import ArgsChecker


class BaseModelABC(ABC):
    @abstractmethod
    @ArgsChecker((None, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series), None)
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        pass

    @abstractmethod
    @ArgsChecker((None, pd.DataFrame), pd.Series)
    def predict(self, X_test: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    @ArgsChecker((None, pd.DataFrame, pd.Series), None)
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        pass
