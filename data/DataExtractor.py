# opti-ml-py\data\DataExtractor.py
import pandas as pd
from typing import Tuple
from decorators.ArgsChecker import ArgsChecker


class DataExtractor:
    @staticmethod
    @ArgsChecker((pd.DataFrame,), tuple)
    def extract_data(
        full_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

        # 'data' 列でフィルタリングし、不要な列を最後に削除
        X_train = full_data[full_data["data"] == "X_train"].drop(columns=["label"])
        X_test = full_data[full_data["data"] == "X_test"].drop(columns=["label"])
        y_train = full_data[full_data["data"] == "X_train"]["label"]
        y_test = full_data[full_data["data"] == "X_test"]["label"]

        # 不要な 'date', 'data', 'symbol' 列を削除
        X_train = X_train.drop(columns=["date", "data", "symbol"])
        X_test = X_test.drop(columns=["date", "data", "symbol"])
        y_train = y_train  # インデックスはオリジナルのまま
        y_test = y_test  # インデックスはオリジナルのまま

        return X_train, X_test, y_train, y_test

    # X_train: トレーニングデータの特徴量
    # X_test: テストデータの特徴量
    # y_train: トレーニングデータのラベル
    # y_test: テストデータのラベル
