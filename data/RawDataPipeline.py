# opti-ml-py\data\DataPipeline.py
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.StockDataFetcherABC import StockDataFetcherABC  # 抽象クラスのインポート
from data.RawDataManager import RawDataManager  # RawDataManager クラスのインポート


class RawDataPipeline:
    @ArgsChecker(
        (None, StockDataFetcherABC, RawDataManager), None
    )  # fetcherがStockDataFetcherABCを継承し、saverがRawDataManagerであるかチェック
    def __init__(self, fetcher: StockDataFetcherABC, saver: RawDataManager):
        self.fetcher = fetcher  # データを取得するオブジェクトを設定
        self.saver = saver  # データを保存するオブジェクトを設定

    @ArgsChecker((None,), None)  # 引数チェックデコレータを適用
    def run(self):
        raw_data = self.fetcher.fetch_data()  # データを取得
        print("Data fetching completed.")  # データ取得完了のメッセージを表示

        standardized_data = self.fetcher.standardize_data(raw_data)  # データを標準化
        print("Data standardization completed.")  # データ標準化完了のメッセージを表示

        if standardized_data.empty:  # 標準化されたデータが空かどうかを確認
            print(
                "No data found for the specified parameters."
            )  # データが見つからなかった場合のメッセージを表示
            return  # 処理を終了

        self.saver.save_raw_data(standardized_data)  # データを保存
        print("Data saving completed.")  # データ保存完了のメッセージを表示

        print(
            "Data pipeline completed successfully."
        )  # パイプライン処理完了のメッセージを表示
