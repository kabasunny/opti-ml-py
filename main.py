import pandas as pd
from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.DataManager import DataManager
from data.RawDataPipeline import RawDataPipeline
from preprocessing.PreprocessPipeline import PreprocessPipeline
from labeling.LabelCreationPipeline import LabelCreationPipeline
from labeling.TroughLabelCreator import TroughLabelCreator
from features.FeaturePipeline import FeaturePipeline
from selectores.SelectorPipeline import SelectorPipeline
from features.AnalyzerFactory import AnalyzerFactory
from selectores.SelectorFactory import SelectorFactory


def main():
    symbol = "7203"
    trade_start_date = pd.Timestamp("2023-08-01")
    before_period_days = 366 * 2  # スタート日より、さかのぼって2年間のデータを取得
    data_start_period = trade_start_date - pd.DateOffset(days=before_period_days)
    end_date = pd.Timestamp("today").strftime("%Y-%m-%d")

    # データ保存ディレクトリのベースパスと拡張子を指定
    base_data_path = "data/stock_data"
    file_ext = "parquet"  # CSVの代わりにparquetを使用 Goで使用可能

    # パス生成用関数
    def generate_path(sub_dir, symbol, end_date, extension):
        return f"{base_data_path}/{sub_dir}/{symbol}_{end_date}.{extension}"

    # 各パイプラインのデータ保存パス
    raw_data_path = generate_path("formated_raw", symbol, end_date, file_ext)
    processed_data_path = generate_path("processed_raw", symbol, end_date, file_ext)
    label_data_path = generate_path("labeled", symbol, end_date, file_ext)
    feature_data_path = generate_path("feature", symbol, end_date, file_ext)
    normalized_f_d_path = generate_path("normalized_ft", symbol, end_date, file_ext)
    selected_f_d_path = generate_path("selected_ft", symbol, end_date, file_ext)

    raw_data_manager = DataManager(raw_data_path)
    processed_d_m = DataManager(processed_data_path)
    label_d_m = DataManager(label_data_path)
    feature_d_m = DataManager(feature_data_path)
    normalize_f_d_m = DataManager(normalized_f_d_path)
    select_f_d_m = DataManager(selected_f_d_path)

    # ----------------------pipeline----------------------
    print("★ DataPipeline ★")
    fetcher = YahooFinanceStockDataFetcher(symbol, data_start_period, end_date)
    RawDataPipeline(fetcher, raw_data_manager).run()
    print("★ PreprocessPipeline ★")
    PreprocessPipeline(raw_data_manager, processed_d_m).run()
    print("★ LabelCreationPipeline ★")
    label_creator = TroughLabelCreator(trade_start_date)
    LabelCreationPipeline(raw_data_manager, label_d_m, label_creator).run()
    print("★ FeatureCreationPipeline ★")
    feature_list_str = ["peak_trough", "fourier", "volume", "price"]  # 特徴量リスト
    analyzers = AnalyzerFactory.create_analyzers(feature_list_str)
    FeaturePipeline(
        processed_d_m,
        feature_d_m,
        normalize_f_d_m,
        analyzers,
        trade_start_date,
    ).run()
    print("★ FeatureSelectionPipeline ★")
    selectors = SelectorFactory.create_selectors()
    SelectorPipeline(label_d_m, normalize_f_d_m, select_f_d_m, selectors).run()


if __name__ == "__main__":
    main()
