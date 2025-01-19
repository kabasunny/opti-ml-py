from models.ModelSaverLoader import ModelSaverLoader
from data.DataManager import DataManager
from models.ModelPredictor import ModelPredictor
from typing import List


class ForRealPredictPipeline:
    def __init__(
        self,
        model_saver_loader: ModelSaverLoader,
        selected_ft_w_l_manager: DataManager,
        bach_predictions_data_manager: DataManager,
        model_types: List[str],
    ):
        self.model_saver_loader = model_saver_loader
        self.selected_feature_manager = selected_ft_w_l_manager
        self.real_predictions_data_manager = bach_predictions_data_manager
        self.model_types = model_types
        self.models = None

    def run(self, symbol):
        # モデルの読み込み
        self.models = self.model_saver_loader.load_models(self.model_types)

        # 実践用データの読み込み
        practical_data = self.selected_feature_manager.load_data(symbol)

        # インデックスでソート
        practical_data = practical_data.sort_values(by="date").reset_index(drop=True)

        # 正解と不正解の数を抽出
        correct_count = practical_data[practical_data["label"] == 1].shape[0]
        incorrect_count = practical_data[practical_data["label"] == 0].shape[0]
        ratio_pr = round(incorrect_count / correct_count, 1)
        print(f"real_prediction by trained model... correct:incorrect = 1:{ratio_pr}")
        print(f"correct:{correct_count}, incorrect:{incorrect_count}")

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
        self.real_predictions_data_manager.save_data(predictions_df, symbol)
        

        # モデルの評価
        evaluations_df = ModelPredictor.evaluate(
            self.models, features, practical_data["label"]
        )

        # 評価結果を出力
        print(evaluations_df)

        
