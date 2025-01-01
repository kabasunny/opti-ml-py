import pandas as pd
from labeling.utils.TroughAndPeakDetector import TroughAndPeakDetector
from labeling.utils.PeriodBasedTroughSelector import PeriodBasedTroughSelector
from labeling.utils.PriceBasedTroughSelector import PriceBasedTroughSelector
from labeling.utils.PeakBasedTroughSelector import PeakBasedTroughSelector
from labeling.LabelCreatorABC import LabelCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class TroughLabelCreator(LabelCreatorABC):
    @ArgsChecker((None, pd.DataFrame), pd.DataFrame)
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        日足の終値が週足トラフと同じ値であるものをラベル付けする関数
        """
        # トラフおよびピークを検出
        troughs = TroughAndPeakDetector.detect_troughs(df["close"])
        peaks = TroughAndPeakDetector.detect_peaks(df["close"])

        # ラベル列を初期化
        df["label"] = 0

        # 日付列をTimestamp型に変換
        df["date"] = pd.to_datetime(df["date"])

        pre_x = 5
        post_x = 20
        high_x = 8.0  # high_x を float 型に変換

        # トラフのインデックスに基づいて選択されたトラフを検出
        selected_period_troughs = (
            PeriodBasedTroughSelector.select_troughs_based_on_priod(
                df, troughs, pre_x, post_x
            )
        )

        selected_price_troughs = PriceBasedTroughSelector.select_troughs_based_on_price(
            df, selected_period_troughs, high_x
        )

        selected_troughs = PeakBasedTroughSelector.select_troughs_based_on_peak(
            df, selected_price_troughs, troughs, peaks, pre_x, high_x
        )

        # ラベルを付ける
        for trough_date in selected_troughs:
            df.loc[df["date"] == trough_date, "label"] = 1

        return df
