# opti-ml-py\models\runner.py
import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from data.DataManager import DataManager
from models.ModelTrainPipeline import ModelTrainPipeline
from models.ModelSaverLoader import ModelSaverLoader
from models.ModelFactory import ModelFactory


def runner():
    # データ保存ディレクトリのベースパスと拡張子を指定
    symbol = "7203"
    base_data_path = "data/stock_data"
    file_ext = "parquet"  # CSVの代わりにparquetを使用
    end_date = pd.Timestamp("today").strftime("%Y-%m-%d")

    model_types = [
        "lightgbm",
        "rand_frst",
        "xgboost",
        "catboost",
        "adaboost",
        "svm",
        "knn",
        "logc_regr",
    ]

    # パス生成用関数
    def generate_path(base_data_path, sub_dir, symbol, end_date, extension):
        return f"{base_data_path}/{sub_dir}/{symbol}_{end_date}.{extension}"

    # 各パイプラインのデータ保存パス
    training_and_test_data_path = generate_path(
        base_data_path, "training_and_test", symbol, end_date, file_ext
    )
    # データマネージャのインスタンスを作成
    training_and_test_data_manager = DataManager(training_and_test_data_path)

    # モデルのインスタンス作成
    models = ModelFactory.create_models(model_types)
    model_save_path = "models/trained_models"
    model_file_ext = "pkl"
    # モデルセーブローダーのインスタンスを作成
    model_saver_loader = ModelSaverLoader(model_save_path, model_file_ext)

    # モデルパイプラインの作成と実行
    ModelTrainPipeline(training_and_test_data_manager, models, model_saver_loader).run()


if __name__ == "__main__":
    runner()
