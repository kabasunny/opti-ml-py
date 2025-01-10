import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PriceBasedTroughSelector:
    @staticmethod
    @ArgsChecker((pd.DataFrame, list, float), list)
    def select_troughs_based_on_price(
        df: pd.DataFrame, selected_period_troughs: list, high_x: float
    ) -> list:
        """
        選択された期間のトラフに基づいて、価格条件を満たすトラフを選択するメソッド
        つまり、一定の利益がの出ないトラフを除外する

        Args:
            df (pd.DataFrame): 株価データフレーム
            selected_period_troughs (list): 選択された期間のトラフの日付リスト
            high_x (float): 終値の上昇率の閾値（パーセント）

        Returns:
            list: 条件を満たすトラフの日付リスト
        """
        selected_troughs = []

        for i in range(len(selected_period_troughs) - 1):
            start_date = selected_period_troughs[i]
            end_date = selected_period_troughs[i + 1]

            # start_dateの日の終値を取得
            start_close = df.loc[df["date"] == start_date, "close"].values[0]
            # end_dateの日の終値を取得
            end_close = df.loc[df["date"] == end_date, "close"].values[0]

            # 終値が開始日（start_date）より低いことを確認
            if end_close > start_close:
                # 期間内の最大終値を取得
                sampling_window = df.loc[
                    (df["date"] > start_date) & (df["date"] <= end_date)
                ]
                max_close_in_period = sampling_window["close"].max()

                # 期間内の終値が閾値（start_close * (1 + high_x / 100)）を超えないことを確認
                if max_close_in_period >= start_close * (1 + high_x / 100):
                    selected_troughs.append(start_date)

        # 最後のトラフは必ず追加
        if selected_period_troughs:
            selected_troughs.append(selected_period_troughs[-1])

        return selected_troughs
