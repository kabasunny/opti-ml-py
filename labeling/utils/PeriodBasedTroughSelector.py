import pandas as pd
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class PeriodBasedTroughSelector:
    @staticmethod
    @ArgsChecker((pd.DataFrame, pd.Series, int, int), list)
    def select_troughs_based_on_priod(
        df: pd.DataFrame, troughs: pd.Series, pre_x: int, post_x: int
    ) -> list:
        """
        トラフのインデックスに基づいて、ラベルを付けるための選択されたトラフを検出するメソッド
        週足ベースで日足がラップするトラフをとらえる

        ただし、この段階では、利益が見込めないトラフを含み、利益が出るトラフを逃すケースが多いと想定される

        Args:
            df (pd.DataFrame): 株価データフレーム
            troughs (pd.Series): トラフのインデックス
            pre_x (int): 前側の検出幅
            post_x (int): 後ろ側の検出幅

        Returns:
            list: 選択されたトラフの日付リスト
        """
        selected_troughs = []

        for i in range(len(troughs)):
            start_idx = max(troughs[i] - pre_x, 0)
            end_idx = min(troughs[i] + post_x + 1, len(df))
            sampling_window = df.iloc[start_idx:end_idx]

            min_close_value = sampling_window["close"].min()
            min_close_date = sampling_window.loc[
                sampling_window["close"] == min_close_value, "date"
            ].values[0]
            min_close_date = pd.Timestamp(min_close_date)

            if selected_troughs:
                last_trough = pd.Timestamp(selected_troughs[-1])

                if last_trough not in df["date"].values:
                    print(
                        f"Warning: last_trough {last_trough} is not in the DataFrame index"
                    )
                    continue

                if min_close_date not in df["date"].values:
                    print(
                        f"Warning: min_close_date {min_close_date} is not in the DataFrame index"
                    )
                    continue

                if (min_close_date - last_trough).days > (pre_x + post_x):
                    selected_troughs.append(min_close_date)
                elif (min_close_date - last_trough).days <= (pre_x + post_x) and (
                    df.loc[df["date"] == last_trough, "close"].values.size > 0
                    and min_close_value
                    < df.loc[df["date"] == last_trough, "close"].values[0]
                ):
                    selected_troughs[-1] = min_close_date
            else:
                selected_troughs.append(min_close_date)

        return selected_troughs
