import pandas as pd

from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher

# from data.RawDataManager import DataManager
# from data.DataManager import DataManager
# from data.DataManager import DataManager
from data.DataManager import DataManager

from data.RawDataPipeline import RawDataPipeline
from preprocessing.PreprocessPipeline import PreprocessPipeline
from labeling.LabelCreationPipeline import LabelCreationPipeline
from labeling.TroughLabelCreator import TroughLabelCreator
from features.FeaturePipeline import FeaturePipeline

from features.AnalyzerFactory import AnalyzerFactory


def main():
    symbol = "7203"
    trade_start_date = pd.Timestamp("2023-08-01")
    before_period_days = 366 * 2  # スタート日より、さかのぼって1年間のデータを取得
    data_start_period = trade_start_date - pd.DateOffset(days=before_period_days)
    end_date = pd.Timestamp("today")

    # API から取得した株価データを保存するファイルパス
    raw_data_path = "data/raw/demo_row_stock_data.csv"
    raw_data_manager = DataManager(raw_data_path)

    # 前処理済みの株価データを保存するファイルパス
    processed_data_path = "data/processed/demo_processed_stock_data.csv"
    processed_data_manager = DataManager(processed_data_path)

    # ラベルデータを保存するファイルパス
    label_data_path = "data/label/demo_labels.csv"
    label_data_manager = DataManager(label_data_path)

    # 特徴量データを保存するファイルパス
    feature_data_path = "data/feature/demo_feature_data.csv"
    feature_data_manager = DataManager(feature_data_path)

    # 正規化済みの特徴量データを保存するファイルパス
    nomalized_feature_data_path = "data/processed/demo_normalized_feature_data.csv"
    normalized_f_d_manager = DataManager(nomalized_feature_data_path)

    print("★ DataPipeline ★")
    fetcher = YahooFinanceStockDataFetcher(symbol, data_start_period, end_date)
    # DataPipeline クラスのインスタンスを作成し、データパイプラインを実行
    RawDataPipeline(fetcher, raw_data_manager).run()

    print("★ PreprocessPipeline ★")
    # PreprocessPipeline のインスタンスを作成し、引数としてデータマネージャを渡す
    PreprocessPipeline(raw_data_manager, processed_data_manager).run()

    print("★ LabelCreationPipeline ★")
    label_creator = TroughLabelCreator(
        trade_start_date
    )  # トラフラベルクリエータークラス
    # LabelCreationPipeline のインスタンスを作成し、実行
    LabelCreationPipeline(raw_data_manager, label_data_manager, label_creator).run()

    print("★ FeaturePipeline ★")
    feature_list_str = ["peak_trough", "fourier", "volume", "price"]  # 特徴量リスト
    analyzers = AnalyzerFactory.create_analyzers(
        feature_list_str
    )  # アナライザーのリストを生成
    # FeaturePipeline のインスタンスを作成し、実行
    FeaturePipeline(
        processed_data_manager,
        feature_data_manager,
        normalized_f_d_manager,
        analyzers,
        trade_start_date,
    ).run()


if __name__ == "__main__":
    main()
