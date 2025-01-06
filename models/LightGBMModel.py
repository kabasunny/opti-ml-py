import lightgbm as lgb
import pandas as pd
from typing import Any, Tuple
from models.BaseModelABC import BaseModelABC
from models.Evaluator import Evaluator
from decorators.ArgsChecker import ArgsChecker


class LightGBMModel(BaseModelABC):
    def __init__(self):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 10,
            "verbose": -1,
        }
        self.model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple["BaseModelABC", Tuple[float, float, float, float]]:
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        self.model = lgb.train(
            self.params,
            lgb_train,
            valid_sets=[lgb_eval],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        result = self.evaluate(X_test, y_test)
        return self, result

    @ArgsChecker((None, pd.DataFrame), pd.Series)
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        predictions = self.model.predict(X_test)
        binary_predictions = (predictions >= 0.5).astype(int)
        return pd.Series(binary_predictions)

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[float, float, float, float]:
        return Evaluator.evaluate_model(self, X_test, y_test)
