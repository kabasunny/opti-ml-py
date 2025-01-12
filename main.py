import pandas as pd
from models.ModelSaverLoader import ModelSaverLoader
from AutomatedPipeline import AutomatedPipeline
from data.DataManager import DataManager


def main():
    symbols = [
        "1570",  # 意図的にエラーを発生するシンボル
        # "7203",  # Toyota Motor Corporation
        # "7201",  # Honda Motor Co., Ltd.
        # "7261",  # Mazda Motor Corporation
        # "7269",  # Suzuki Motor Corporation
        # "7270",  # Subaru Corporation
        # "7202",  # Isuzu Motors Limited
        # "7205",  # Hino Motors, Ltd.
        # "7224",  # Shizuoka Daihatsu Motor Co., Ltd.
    ]

    before_period_days = 365 * 2  # 特徴量生成に必要なデータ期間
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
    file_ext = "csv"  # "parquet"

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
    for d_m_name in data_manager_names:
        data_managers[d_m_name] = DataManager(base_data_path, d_m_name, file_ext)

    pipeline = AutomatedPipeline(
        before_period_days,
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
