import pandas as pd
from models.ModelSaverLoader import ModelSaverLoader
from data.DataManager import DataManager
from models.ModelPredictor import ModelPredictor
from typing import List


class ModelPredictPipeline:
    def __init__(
        self,
        model_saver_loader: ModelSaverLoader,
        practical_data_manager: DataManager,
        predictions_data_manager: DataManager,
        model_types: List[str],
    ):
        self.model_saver_loader = model_saver_loader
        self.practical_data_manager = practical_data_manager
        self.predictions_data_manager = predictions_data_manager
        self.model_types = model_types

    def run(self):
        # モデルの読み込み
        models = self.model_saver_loader.load_models(self.model_types)

        # 実践用データの読み込み
        practical_data = self.practical_data_manager.load_data()

        # インデックスでソート
        practical_data = practical_data.sort_values(by="date").reset_index(drop=True)
        # print(practical_data.head(30))
        # print(len(practical_data))

        # 特徴量を抽出
        features = practical_data.drop(
            columns=["date", "symbol", "label"]
        )  # 必要に応じて列を調整
        # print(len(features))

        # モデルによる予測
        predictions_df = ModelPredictor.predict(models, features)
        # print(len(predictions_df))

        # インデックスをリセットしてから結合する
        predictions_df = predictions_df.reset_index(drop=True)

        # 予測結果をDataFrameにまとめて保存
        predictions_df["date"] = practical_data["date"]
        predictions_df["symbol"] = practical_data["symbol"]
        predictions_df["label"] = practical_data["label"]
        self.predictions_data_manager.save_data(predictions_df)

        # モデルの評価
        evaluations_df = ModelPredictor.evaluate(
            models, features, practical_data["label"]
        )
        # print(len(evaluations_df))

        # 評価結果を出力
        print(evaluations_df)
