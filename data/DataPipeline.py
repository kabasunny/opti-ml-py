from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class DataPipeline:
    @ArgsChecker((None, object, object), None)
    def __init__(self, fetcher, saver):
        self.fetcher = fetcher
        self.saver = saver

    @ArgsChecker(
        (
            None,
            str,
        ),
        None,
    )
    def run(self, save_path):
        print("Fetching data...")
        raw_data = self.fetcher.fetch_data()
        print("Standardizing data...")
        standardized_data = self.fetcher.standardize_data(raw_data)
        if standardized_data.empty:
            print("No data found for the specified parameters.")
            return
        print("Saving data...")
        self.saver.save_raw_data(standardized_data, save_path)
        print("Data pipeline completed successfully.")
