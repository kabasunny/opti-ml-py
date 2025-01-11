import pandas as pd
from models.ModelSaverLoader import ModelSaverLoader
from AutomatedPipeline import AutomatedPipeline
from data.DataManager import DataManager


def main():
    symbols = [
        # "1570",  # 意図的にエラーを発生するシンボル
        "7203",  # Toyota Motor Corporation
        "7201",  # Honda Motor Co., Ltd.
        "7261",  # Mazda Motor Corporation
        "7269",  # Suzuki Motor Corporation
        "7270",  # Subaru Corporation
        "7202",  # Isuzu Motors Limited
        "7205",  # Hino Motors, Ltd.
        "7224",  # Shizuoka Daihatsu Motor Co., Ltd.
    ]

    trade_start_date = pd.Timestamp("2003-08-01")
    before_period_days = 366 * 2  # スタート日より、さかのぼって2年間のデータを取得
    data_start_period = trade_start_date - pd.DateOffset(days=before_period_days)
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

    model_saver_loader = ModelSaverLoader(
        model_save_path="models/trained_models", model_file_ext="pkl"
    )

    feature_list_str = ["peak_trough", "fourier", "volume", "price", "past"]

    base_data_path = "data/stock_data"
    file_ext = "parquet"

    data_manager_names = [
        "formated_raw",
        "processed_raw",
        "labeled",
        "normalized_feature",
        "selected_feature",
        "training_and_test",
        "practical",
        "predictions",
    ]

    data_managers = {}
    for name in data_manager_names:
        data_managers[name] = DataManager(base_data_path, file_ext, name, end_date)

    pipeline = AutomatedPipeline(
        trade_start_date,
        data_start_period,
        end_date,
        model_types,
        feature_list_str,
        model_saver_loader,
        data_managers,
    )

    while symbols:
        symbol = symbols.pop(0)
        pipeline.process_symbol(symbol)


if __name__ == "__main__":
    main()
