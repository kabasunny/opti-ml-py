import pandas as pd
from models.ModelSaverLoader import ModelSaverLoader
from data.DataManager import DataManager
from models.ModelPredictor import ModelPredictor
from data.DataExtractor import DataExtractor
from typing import List
from data.DataForModelPreparation import DataPreparation
from models.ModelTrainer import ModelTrainer


class ModelPredictPipeline:
    def __init__(
        self,
        model_saver_loader: ModelSaverLoader,
        training_and_test_manager: DataManager,  # 追加トレーニング 非学習用廃棄データを再利用して学習
        practical_data_manager: DataManager,
        predictions_data_manager: DataManager,
        model_types: List[str],
    ):
        self.model_saver_loader = model_saver_loader
        self.training_and_test_manager = training_and_test_manager  # 追加トレーニング 非学習用廃棄データを再利用して学習
        self.practical_data_manager = practical_data_manager
        self.predictions_data_manager = predictions_data_manager
        self.model_types = model_types
        self.models = None

    def run(self):
        # モデルの読み込み
        self.models = self.model_saver_loader.load_models(self.model_types)

        # 実践用データの読み込み
        practical_data = self.practical_data_manager.load_data()

        # インデックスでソート
        practical_data = practical_data.sort_values(by="date").reset_index(drop=True)

        # 特徴量を抽出
        features = practical_data.drop(
            columns=["date", "symbol", "label"]
        )  # 必要に応じて列を調整

        # モデルによる予測
        predictions_df = ModelPredictor.predict(self.models, features)

        # インデックスをリセットしてから結合する
        predictions_df = predictions_df.reset_index(drop=True)

        # 予測結果をDataFrameにまとめて保存
        predictions_df["date"] = practical_data["date"]
        predictions_df["symbol"] = practical_data["symbol"]
        predictions_df["label"] = practical_data["label"]
        self.predictions_data_manager.save_data(predictions_df)

        # モデルの評価
        evaluations_df = ModelPredictor.evaluate(
            self.models, features, practical_data["label"]
        )

        # 評価結果を出力
        print(evaluations_df)

        # 追加トレーニング 非学習用廃棄データを再利用して学習

        rate = 3
        print(f"Additional training... \ncorrect : incorrect = 1 : {rate}")
        _, re_featuer, _, re_label = DataExtractor.extract_data(
            self.training_and_test_manager.load_data()
        )

        # 再利用データを結合
        re_featuer = pd.concat([re_featuer, features], ignore_index=True)
        re_label = pd.concat([re_label, practical_data["label"]], ignore_index=True)
        #
        # full_dataの作成
        full_data = re_featuer.copy()
        full_data["label"] = re_label

        correct_data, incorrect_data = DataPreparation.split_data_by_label(full_data)

        # correct_data : incorrect_data = 1 : rate の比率にするため、incorrect_dataの数を調整
        desired_incorrect_size = rate * correct_data.shape[0]
        if incorrect_data.shape[0] > desired_incorrect_size:
            incorrect_data = incorrect_data.sample(
                n=desired_incorrect_size, random_state=42
            )

        X_train, X_test, y_train, y_test = DataPreparation.add_training_and_test_data(
            correct_data, incorrect_data
        )

        self.models, results_df = ModelTrainer.train(
            self.models, X_train, y_train, X_test, y_test
        )
        print(results_df)

        self.model_saver_loader.save_models(self.models)
