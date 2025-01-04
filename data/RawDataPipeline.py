# opti-ml-py\data\DataPipeline.py
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.StockDataFetcherABC import StockDataFetcherABC  # 抽象クラスのインポート
from data.DataManager import DataManager  # DataManager クラスのインポート


class RawDataPipeline:
    @ArgsChecker(
        (None, StockDataFetcherABC, DataManager), None
    )  # fetcherがStockDataFetcherABCを継承し、saverがRawDataManagerであるかチェック
    def __init__(self, fetcher: StockDataFetcherABC, raw_data_manager: DataManager):
        self.fetcher = fetcher  # データを取得するオブジェクトを設定
        self.raw_data_manager = raw_data_manager  # データを保存するオブジェクトを設定

    @ArgsChecker((None,), None)  # 引数チェックデコレータを適用
    def run(self):
        print("Run Data pipeline completed successfully")
        raw_data = self.fetcher.fetch_data()  # データを取得
        # print("Data fetching completed.")  # データ取得完了のメッセージを表示

        standardized_data = self.fetcher.format_data(raw_data)  # データを標準化
        # print("Data standardization completed.")  # データ標準化完了のメッセージを表示

        if standardized_data.empty:  # 標準化されたデータが空かどうかを確認
            print("No data found for the specified parameters")
            return  # 処理を終了

        self.raw_data_manager.save_data(standardized_data)  # データを保存
        # print("Data saving completed")  # データ保存完了のメッセージを表示

        print("Data pipeline completed successfully")
