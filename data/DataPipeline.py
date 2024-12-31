from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.StockDataFetcherABC import StockDataFetcherABC  # 抽象クラスのインポート
from data.DataSaver import DataSaver  # 抽象クラスのインポート
from data.DataPipeline import DataPipeline  # 抽象クラスのインポート


class DataPipeline:
    @ArgsChecker(
        (None, StockDataFetcherABC, DataSaver), None
    )  # fetcherがStockDataFetcherABCを継承し、saverがDataSaverABCであるかチェック
    def __init__(self, fetcher: StockDataFetcherABC, saver: DataSaver):
        self.fetcher = fetcher  # データを取得するオブジェクトを設定
        self.saver = saver  # データを保存するオブジェクトを設定

    @ArgsChecker((None, str), None)  # 引数チェックデコレータを適用
    def run(self, save_path: str):
        print("Fetching data...")  # データ取得開始のメッセージを表示
        raw_data = self.fetcher.fetch_data()  # データを取得
        print("Standardizing data...")  # データ標準化開始のメッセージを表示
        standardized_data = self.fetcher.standardize_data(raw_data)  # データを標準化
        if standardized_data.empty:  # 標準化されたデータが空かどうかを確認
            print(
                "No data found for the specified parameters."
            )  # データが見つからなかった場合のメッセージを表示
            return  # 処理を終了
        print("Saving data...")  # データ保存開始のメッセージを表示
        self.saver.save_raw_data(standardized_data, save_path)  # データを保存
        print(
            "Data pipeline completed successfully."
        )  # パイプライン処理完了のメッセージを表示
