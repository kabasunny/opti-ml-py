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
        # # デバッグ用出力
        # print("Original full_data:\n", full_data.head())

        # 'data' 列でフィルタリングし、不要な列を最後に削除
        X_train = full_data[full_data["data"] == "X_train"].drop(columns=["label"])
        X_test = full_data[full_data["data"] == "X_test"].drop(columns=["label"])
        y_train = full_data[full_data["data"] == "X_train"]["label"]
        y_test = full_data[full_data["data"] == "X_test"]["label"]

        # # デバッグ用出力
        # print("\nFiltered X_train:\n", X_train.head())
        # print("\nFiltered X_test:\n", X_test.head())
        # print("\ny_train:\n", y_train.head())
        # print("\ny_test:\n", y_test.head())

        # 不要な 'date', 'data', 'symbol' 列を削除
        X_train = X_train.drop(columns=["date", "data", "symbol"])
        X_test = X_test.drop(columns=["date", "data", "symbol"])
        y_train = y_train  # インデックスはオリジナルのまま
        y_test = y_test  # インデックスはオリジナルのまま

        # # デバッグ用出力
        # print("\nFinal X_train:\n", X_train.head())
        # print("\nFinal X_test:\n", X_test.head())
        # print("\nFinal y_train:\n", y_train.head())
        # print("\nFinal y_test:\n", y_test.head())

        return X_train, X_test, y_train, y_test
