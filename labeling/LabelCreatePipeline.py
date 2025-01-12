import pandas as pd
from decorators.ArgsChecker import ArgsChecker
from data.DataManager import DataManager
from labeling.LabelCreatorABC import LabelCreatorABC


class LabelCreatePipeline:
    @ArgsChecker((None, DataManager, DataManager, int, LabelCreatorABC), None)
    def __init__(
        self,
        raw_data_manager,
        label_data_manager,
        before_period_days,
        label_creator,
    ):
        self.raw_data_manager = raw_data_manager
        self.label_data_manager = label_data_manager
        self.before_period_days = before_period_days
        self.label_creator = label_creator

    @ArgsChecker((None, str), None)
    def run(self, symbol):
        """データパイプラインの実行"""
        # データの読み込み
        df = self.raw_data_manager.load_data(symbol)

        # trade_start_date を計算
        first_date = pd.to_datetime(df["date"].iloc[0])
        trade_start_date = first_date + pd.DateOffset(days=self.before_period_days)

        # ラベルの作成
        labels = self.label_creator.create_labels(df, trade_start_date)
        # print(f"labels\n{labels.head(1)}")

        # ラベルデータの保存
        self.label_data_manager.save_data(labels, symbol)

        print("Label creation pipeline completed successfully")
