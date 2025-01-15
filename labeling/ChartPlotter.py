import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import matplotlib.pyplot as plt
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート
from data.DataManager import DataManager  # DataManager クラスをインポート


class ChartPlotter:
    @staticmethod
    @ArgsChecker((pd.DataFrame, pd.DataFrame, str), None)
    def plot_data(base_df: pd.DataFrame, label_df: pd.DataFrame, symbol: str):
        """ラベルデータをチャートにプロットするメソッド"""
        try:
            # 日付をdatetime型に変換
            base_df["date"] = pd.to_datetime(base_df["date"])
            label_df["date"] = pd.to_datetime(label_df["date"])

            plt.figure(figsize=(14, 7))  # プロットのサイズを設定
            plt.plot(
                base_df["date"], base_df["close"], label="Close Price", color="blue"
            )  # 終値をプロット

            # ベースデータとラベルデータを日付でマージ
            merged_df = pd.merge(
                base_df,
                label_df[label_df["label"] == 1][["date", "label"]],
                on="date",
                how="left",
            )

            # ラベルが1のポイントをプロット
            plt.scatter(
                merged_df[merged_df["label"] == 1]["date"],
                merged_df[merged_df["label"] == 1]["close"],
                label="Label",
                color="red",
                marker="o",
            )

            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.title(f"{symbol} Close Price with Labels")
            plt.legend()  # 凡例を表示

            # 表示期間を計算
            period = (base_df["date"].max() - base_df["date"].min()).days

            # x軸の目盛りを年単位または月単位に設定し、フォーマットを適用
            ax = plt.gca()
            if period > 3650:  # 10年以上の場合
                ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
            else:
                ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
                ax.xaxis.set_major_formatter(
                    plt.matplotlib.dates.DateFormatter("%Y-%m")
                )

            plt.xticks(rotation=45)  # x軸の目盛りを45度回転
            plt.grid(True)  # グリッドを表示
            plt.tight_layout()  # レイアウトを調整

            # インセットプロット（小窓）を作成
            inset_ax = plt.axes([0.65, 0.55, 0.2, 0.3])  # [left, bottom, width, height]
            labels = ["Close", "Label"]
            close_count = base_df["close"].count()
            label_count = label_df[label_df["label"] == 1]["label"].count()
            counts = [close_count, label_count]
            bars = inset_ax.bar(labels, counts, color=["blue", "red"])
            inset_ax.set_title("Counts")
            inset_ax.set_ylabel("Number of Points")

            # バーの上に数値を表示
            for bar in bars:
                height = bar.get_height()
                inset_ax.annotate(
                    "{}".format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

            plt.show()  # プロットを表示
        except Exception as e:
            print(f"チャートプロットに失敗しました: {e}")


# 使用例
if __name__ == "__main__":
    symbol = "6861"
    base_data_path = "data/stock_data"
    file_ext = "csv"  # parquet

    # DataManager クラスを使用してデータをロード
    manager_base = DataManager(base_data_path, "formated_raw", file_ext)
    manager_label = DataManager(base_data_path, "labeled", file_ext)

    # ベースデータとラベルデータをロード
    base_df = manager_base.load_data(symbol)
    label_df = manager_label.load_data(symbol)

    # チャートをプロット
    ChartPlotter.plot_data(base_df, label_df, symbol)
