import pandas as pd

from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.RawDataManager import RawDataManager
from data.ProcessedDataManager import ProcessedDataManager
from data.LabelDataManager import LabelDataManager

from data.RawDataPipeline import RawDataPipeline
from preprocessing.PreprocessPipeline import PreprocessPipeline
from labeling.LabelCreationPipeline import LabelCreationPipeline
from labeling.TroughLabelCreator import TroughLabelCreator


def main():
    symbol = "7203"
    trade_start_date = pd.Timestamp("2023-08-01")
    before_period_days = 366 * 2  # スタート日より、さかのぼって1年間のデータを取得
    data_start_period = trade_start_date - pd.DateOffset(days=before_period_days)
    end_date = pd.Timestamp("today")

    raw_data_path = "data/raw/demo_row_stock_data.csv"
    raw_data_manager = RawDataManager(raw_data_path)

    processed_data_path = "data/processed/demo_processed_stock_data.csv"
    processed_data_manager = ProcessedDataManager(processed_data_path)

    label_data_path = "data/label/demo_labels.csv"
    label_data_manager = LabelDataManager(label_data_path)

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


if __name__ == "__main__":
    main()
