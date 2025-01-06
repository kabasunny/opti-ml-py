# opti-ml-py\models\BaseModelABC.py
from __future__ import annotations  # 型ヒントの評価を遅延

from abc import ABC, abstractmethod
from typing import Any, Tuple
import pandas as pd
from decorators.ArgsChecker import ArgsChecker


class BaseModelABC(ABC):

    @abstractmethod
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
        pass

    @abstractmethod
    # @ArgsChecker((None, pd.DataFrame), pd.Series)
    def predict(self, X_test: pd.DataFrame) -> Any:
        pass

    @abstractmethod
    # @ArgsChecker((None, pd.DataFrame, pd.Series), Tuple[float, float, float, float])
    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[float, float, float, float]:
        pass
