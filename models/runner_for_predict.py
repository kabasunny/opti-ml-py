# opti-ml-py\models\predict_runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from data.DataManager import DataManager
from models.ModelPredictPipeline import ModelPredictPipeline
from models.ModelSaverLoader import ModelSaverLoader


def runner():
    # データ保存ディレクトリのベースパスと拡張子を指定
    symbol = "7203"
    base_data_path = "data/stock_data"
    file_ext = "parquet"  # CSVの代わりにparquetを使用
    end_date = pd.Timestamp("today").strftime("%Y-%m-%d")

    model_types = [
        "LightGBM",
        "RandomForest",
        "XGBoost",
        "CatBoost",
        "AdaBoost",
        "SVM",
        "KNeighbors",
        "LogisticRegression",
    ]

    # パス生成用関数
    def generate_path(base_data_path, sub_dir, symbol, end_date, extension):
        return f"{base_data_path}/{sub_dir}/{symbol}_{end_date}.{extension}"

    # 実践用データの保存パス
    practical_data_path = generate_path(
        base_data_path, "practical", symbol, end_date, file_ext
    )

    # 予測結果の保存パス
    predictions_save_path = generate_path(
        base_data_path, "predictions", symbol, end_date, file_ext
    )

    # モデルセーブローダーのインスタンスを作成
    model_save_path = "models/trained_models"
    model_file_ext = "pkl"
    model_saver_loader = ModelSaverLoader(model_save_path, model_file_ext)

    # データマネージャのインスタンスを作成
    practical_data_manager = DataManager(practical_data_path)
    predictions_data_manager = DataManager(predictions_save_path)

    # 予測パイプラインの作成と実行
    predict_pipeline = ModelPredictPipeline(
        model_saver_loader,
        practical_data_manager,
        predictions_data_manager,
        model_types,
    )
    predict_pipeline.run()


if __name__ == "__main__":
    runner()
