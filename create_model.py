import pandas as pd
from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.DataManager import DataManager
from data.RawDataPipeline import RawDataPipeline
from preprocessing.PreprocessPipeline import PreprocessPipeline
from labeling.LabelCreationPipeline import LabelCreationPipeline
from labeling.TroughLabelCreator import TroughLabelCreator
from features.FeaturePipeline import FeaturePipeline
from selectores.SelectorPipeline import SelectorPipeline
from data.DataForModelPipeline import DataForModelPipeline
from features.AnalyzerFactory import AnalyzerFactory
from selectores.SelectorFactory import SelectorFactory
from models.ModelTrainPipeline import ModelTrainPipeline
from models.ModelSaverLoader import ModelSaverLoader
from models.ModelFactory import ModelFactory
from models.ModelPredictPipeline import ModelPredictPipeline


def main():

    # 学習シミュレーション条件
    symbol = "7203"
    trade_start_date = pd.Timestamp("2003-08-01")
    before_period_days = 366 * 2  # スタート日より、さかのぼって2年間のデータを取得
    data_start_period = trade_start_date - pd.DateOffset(days=before_period_days)
    end_date = pd.Timestamp("today").strftime("%Y-%m-%d")

    # ----------------------data----------------------
    # データ保存ディレクトリのベースパスと拡張子を指定
    base_data_path = "data/stock_data"
    file_ext = "parquet"  # CSVの代わりにparquetを使用 Goで使用可能

    # 各パイプラインのデータ保存パス作成
    def generate_path(data_name):
        return f"{base_data_path}/{data_name}/{symbol}_{end_date}.{file_ext}"

    raw_data_path = generate_path("formated_raw")
    processed_data_path = generate_path("processed_raw")
    label_data_path = generate_path("labeled")
    feature_data_path = generate_path("feature")
    normalized_f_d_path = generate_path("normalized_ft")
    selected_f_d_path = generate_path("selected_ft")
    training_test_d_p = generate_path("training_and_test")
    practical_d_p = generate_path("practical")
    predictions_save_path = generate_path("predictions")

    # データマネージャのインスタンスを作成
    raw_data_manager = DataManager(raw_data_path)
    prsd_d_m = DataManager(processed_data_path)
    l_d_m = DataManager(label_data_path)
    f_d_m = DataManager(feature_data_path)
    n_f_d_m = DataManager(normalized_f_d_path)
    s_f_d_m = DataManager(selected_f_d_path)
    tr_tt_d_m = DataManager(training_test_d_p)
    prct_d_m = DataManager(practical_d_p)
    pred_d_m = DataManager(predictions_save_path)

    # ----------------------model----------------------
    # 学習用モデル
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
    models = ModelFactory.create_models(model_types)
    model_save_path = "models/trained_models"
    model_file_ext = "pkl"
    # モデルセーブローダーのインスタンスを作成
    model_saver_loader = ModelSaverLoader(model_save_path, model_file_ext)

    # ----------------------pipeline----------------------
    print("★ DataPipeline ★")
    fetcher = YahooFinanceStockDataFetcher(symbol, data_start_period, end_date)
    RawDataPipeline(fetcher, raw_data_manager).run()
    print("★ PreprocessPipeline ★")
    PreprocessPipeline(raw_data_manager, prsd_d_m).run()
    print("★ LabelCreationPipeline ★")
    label_creator = TroughLabelCreator(trade_start_date)
    LabelCreationPipeline(raw_data_manager, l_d_m, label_creator).run()
    print("★ FeatureCreationPipeline ★")
    feature_list_str = ["peak_trough", "fourier", "volume", "price"]  # 特徴量リスト
    analyzers = AnalyzerFactory.create_analyzers(feature_list_str)
    FeaturePipeline(
        prsd_d_m,
        f_d_m,
        n_f_d_m,
        analyzers,
        trade_start_date,
    ).run()
    print("★ FeatureSelectionPipeline ★")
    selectors = SelectorFactory.create_selectors()
    SelectorPipeline(l_d_m, n_f_d_m, s_f_d_m, selectors).run()
    print("★ DataForModelPipeline ★")
    DataForModelPipeline(
        l_d_m,
        f_d_m,
        s_f_d_m,
        tr_tt_d_m,
        prct_d_m,
    ).run()
    print("★ ModelTrainPipeline ★")
    ModelTrainPipeline(tr_tt_d_m, models, model_saver_loader).run()
    print("★ ModelPredictPipeline ★")
    ModelPredictPipeline(
        model_saver_loader,
        prct_d_m,
        pred_d_m,
        model_types,
    ).run()


if __name__ == "__main__":
    main()
