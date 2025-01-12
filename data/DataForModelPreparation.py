import pandas as pd
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager
from sklearn.model_selection import train_test_split


class DataForModelPreparation:
    @staticmethod
    @ArgsChecker((DataManager, DataManager, str), pd.DataFrame)  # 型チェックを追加
    def create_full_data(label_data_manager, feature_data_manager, symbol):

        # ラベルデータを読み込み、label列のみを付加する
        label_data = label_data_manager.load_data(symbol)[["date", "symbol", "label"]]

        # 特徴量データを読み込む
        feature_data = feature_data_manager.load_data(symbol)

        # データを結合する
        full_data = feature_data.merge(label_data, on=["date", "symbol"], how="left")
        # full_data = full_data.merge(feature_data, on=["date", "symbol"], how="left")

        return full_data

    @staticmethod
    @ArgsChecker((pd.DataFrame,), (pd.DataFrame, pd.DataFrame))  # 戻り値の型を修正
    def split_data_by_label(full_data):
        # 不正解データと正解データに分割する
        correct_data = full_data[full_data["label"] == 1]
        incorrect_data = full_data[full_data["label"] == 0]

        return correct_data, incorrect_data

    @staticmethod
    @ArgsChecker(
        (pd.DataFrame, pd.DataFrame),
        (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame),
    )
    def split_data_for_modeling(correct_data, incorrect_data):
        # コレクトデータをランダムに二つに分割
        correct_data_train_eval = correct_data.sample(frac=0.5, random_state=42)
        correct_data_practical_test = correct_data.drop(correct_data_train_eval.index)

        # 不正解データをランダムに分割
        # CASE1:正解と同じ数
        # incorrect_data_train_eval = incorrect_data.sample(
        #     n=correct_data_train_eval.shape[0], random_state=42
        # )
        # CASE2:全数の半分の数
        incorrect_data_train_eval = incorrect_data.sample(frac=0.5, random_state=42)

        incorrect_data_practical_test = incorrect_data.drop(
            incorrect_data_train_eval.index
        )
        # print(f"correct_data_train_eval:{len(correct_data_train_eval)}")
        # print(f"correct_data_practical_test:{len(correct_data_practical_test)}")
        # print(f"incorrect_data_train_eval:{len(incorrect_data_train_eval)}")
        # print(f"incorrect_data_practical_test:{len(incorrect_data_practical_test)}")
        return (
            correct_data_train_eval,
            correct_data_practical_test,
            incorrect_data_train_eval,
            incorrect_data_practical_test,
        )

    @staticmethod
    @ArgsChecker((pd.DataFrame, pd.DataFrame), pd.DataFrame)
    def prepare_training_and_test_data(
        correct_data_train_eval, incorrect_data_train_eval
    ):
        # 訓練データとテストデータに分割
        combined_train_eval_data = pd.concat(
            [correct_data_train_eval, incorrect_data_train_eval]
        )
        # 特徴量とラベルに分ける
        X = combined_train_eval_data.drop(columns=["label"])
        y = combined_train_eval_data["label"]
        # データを訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        # 訓練データに 'data' 列を追加
        train_data = X_train.copy()
        train_data["label"] = y_train
        train_data["data"] = "X_train"
        # テストデータに 'data' 列を追加
        test_data = X_test.copy()
        test_data["label"] = y_test
        test_data["data"] = "X_test"
        # 訓練データとテストデータを結合
        combined_data = pd.concat([train_data, test_data], axis=0)

        return combined_data

    @staticmethod
    @ArgsChecker((pd.DataFrame, pd.DataFrame), pd.DataFrame)  # 型チェックを追加
    def prepare_practical_data(
        correct_data_practical_test, incorrect_data_practical_test
    ):
        # 実践テストデータを合成
        practical_data = pd.concat(
            [correct_data_practical_test, incorrect_data_practical_test]
        )
        practical_data["label"] = 1
        practical_data.loc[
            practical_data.index.isin(incorrect_data_practical_test.index), "label"
        ] = 0
        # 'label' 列を一番右に移動
        practical_data = practical_data[
            [col for col in practical_data.columns if col != "label"] + ["label"]
        ]
        return practical_data

    @staticmethod
    @ArgsChecker((pd.DataFrame, pd.DataFrame), tuple)
    def add_training_and_test_data(correct_data_train_eval, incorrect_data_train_eval):
        # 訓練データとテストデータに分割
        combined_train_eval_data = pd.concat(
            [correct_data_train_eval, incorrect_data_train_eval]
        )
        # 特徴量とラベルに分ける
        X = combined_train_eval_data.drop(columns=["label"])
        y = combined_train_eval_data["label"]
        # データを訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.1,
            shuffle=True,
            random_state=42,
        )

        return X_train, X_test, y_train, y_test
