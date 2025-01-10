import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PeakBasedTroughSelector:
    @staticmethod
    @ArgsChecker((pd.DataFrame, list, pd.Series, pd.Series, int, float), list)
    def select_troughs_based_on_peak(
        df: pd.DataFrame,
        selected_price_troughs: list,
        troughs: pd.Series,
        peaks: pd.Series,
        pre_x: int,
        high_x: float,
    ) -> list:
        """
        ピークの日付より pre_x 日前以内のトラフが、( peak - trough ) / trough > high_x のとき、
        selected_price_troughs に追加するメソッド。ただし、重複させない
        つまり、ピークから逆算して、一定の利益が出るトラフを追加する

        Args:
            df (pd.DataFrame): 株価データフレーム
            selected_price_troughs (list): 選択された価格条件に基づくトラフの日付リスト
            troughs (pd.Series): トラフのインデックス
            peaks (pd.Series): ピークのインデックス
            pre_x (int): ピークからの前側の日数
            high_x (float): ピークとトラフの比率条件

        Returns:
            list: 最終的に条件を満たすトラフの日付リスト
        """
        final_selected_troughs = set(
            selected_price_troughs
        )  # 重複を避けるためにセットを使用

        for peak_idx in peaks:
            peak_date = df.iloc[peak_idx]["date"]
            peak_close = df.iloc[peak_idx]["close"]

            # pre_x 日前までの期間のトラフを取得
            start_idx = max(peak_idx - pre_x, 0)
            pre_x_period = df.iloc[start_idx:peak_idx]

            for _, row in pre_x_period.iterrows():
                trough_close = row["close"]
                trough_date = row["date"]

                # 条件を満たす場合
                if (peak_close - trough_close) / trough_close > high_x / 100:
                    final_selected_troughs.add(trough_date)

        return list(final_selected_troughs)  # リストに変換して返す
