# opti-ml-py\data\DataPipeline.py
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.StockDataFetcherABC import StockDataFetcherABC  # 抽象クラスのインポート
from data.DataManager import DataManager  # DataManager クラスのインポート


class RawDataPipeline:
    @ArgsChecker(
        (None, DataManager, StockDataFetcherABC), None
    )  # fetcherがStockDataFetcherABCを継承し、saverがRawDataManagerであるかチェック
    def __init__(self, raw_data_manager: DataManager, fetcher: StockDataFetcherABC):
        self.fetcher = fetcher  # データを取得するオブジェクトを設定
        self.raw_data_manager = raw_data_manager  # データを保存するオブジェクトを設定

    @ArgsChecker((None, str), None)  # 引数チェックデコレータを適用
    def run(self, symbol):
        # print("Run Data pipeline")
        raw_data = self.fetcher.fetch_data(symbol)  # データを取得
        # print("Data fetching completed.")  # データ取得完了のメッセージを表示
        # print(f"raw_data{len(raw_data)}")

        formated_data = self.fetcher.format_data(
            raw_data, symbol
        )  # データをフォーマット
        # print("Data standardization completed.")  # データ標準化完了のメッセージを表示

        if formated_data.empty:  # 標準化されたデータが空かどうかを確認
            print("No data found for the specified parameters")
            return  # 処理を終了

        self.raw_data_manager.save_data(formated_data, symbol)  # データを保存
        # print("Data saving completed")  # データ保存完了のメッセージを表示

        print("Data pipeline completed successfully")
