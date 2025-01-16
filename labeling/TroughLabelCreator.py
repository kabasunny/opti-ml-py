import pandas as pd
from labeling.utils.TroughAndPeakDetector import TroughAndPeakDetector
from labeling.utils.PeriodBasedTroughSelector import PeriodBasedTroughSelector
from labeling.utils.PriceBasedTroughSelector import PriceBasedTroughSelector
from labeling.utils.PeakBasedTroughSelector import PeakBasedTroughSelector
from labeling.LabelCreatorABC import LabelCreatorABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート

class TroughLabelCreator(LabelCreatorABC):
    def __init__(
        self,
    ):
        pass

    @ArgsChecker((None, pd.DataFrame, pd.Timestamp), pd.DataFrame)
    def create_labels(
        self, df: pd.DataFrame, trade_start_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        週足ベースで底値付近を正解
        ピークから逆算し、ある程度の値幅がある場合に正解
        """
        # 日付列をTimestamp型に変換
        df["date"] = pd.to_datetime(df["date"])

        # trade_start_date 以降の日付のデータをフィルタリング
        df = df[df["date"] >= trade_start_date].copy()
        # print(f"df1\n{df.head(1)}")

        # トラフおよびピークを検出
        troughs = TroughAndPeakDetector.detect_troughs(df["close"])
        peaks = TroughAndPeakDetector.detect_peaks(df["close"])

        # ラベル列を初期化
        df.loc[:, "label"] = 0

        pre_x = 5
        post_x = 20
        high_x = 10.0  # high_x を float 型に変換

        # トラフのインデックスに基づいて選択されたトラフを検出
        selected_period_troughs = (
            PeriodBasedTroughSelector.select_troughs_based_on_priod(
                df, troughs, pre_x, post_x
            )
        )

        # 一定の利益がの出ないトラフを除外する
        selected_price_troughs = PriceBasedTroughSelector.select_troughs_based_on_price(
            df, selected_period_troughs, high_x
        )

        # ピークから逆算して、一定の利益が出るトラフを追加する
        selected_troughs = PeakBasedTroughSelector.select_troughs_based_on_peak(
            df, selected_price_troughs, troughs, peaks, pre_x, high_x
        )

        # ラベルを付ける
        for trough_date in selected_troughs:
            df.loc[df["date"] == trough_date, "label"] = 1

            # 正解ラベルの日付の前日および翌日もラベルとして設定
            for offset in range(-2, 3):  # 二日前から二日後まで
                if offset != 0:
                    adjusted_date = trough_date + pd.Timedelta(days=offset)
                    df.loc[df["date"] == adjusted_date, "label"] = 1

        # データの最後の post_x 日分を不正解ラベルとする
        invalid_label_start_date = df["date"].iloc[-post_x]
        df.loc[df["date"] >= invalid_label_start_date, "label"] = 0

        # 指定された列を削除
        columns_to_drop = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df.drop(columns=columns_to_drop, inplace=True)
        # print(f"df2\n{df.head(1)}")
        return df
