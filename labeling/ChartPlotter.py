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
from data.LabelDataManager import LabelDataManager


class ChartPlotter:
    @staticmethod
    @ArgsChecker((pd.DataFrame,), None)
    def plot_data(df: pd.DataFrame):
        """ラベルデータをチャートにプロットするメソッド"""
        try:
            # 日付をdatetime型に変換
            df["date"] = pd.to_datetime(df["date"])

            plt.figure(figsize=(14, 7))  # プロットのサイズを設定
            plt.plot(
                df["date"], df["close"], label="Close Price", color="blue"
            )  # 終値をプロット
            plt.scatter(
                df[df["label"] == 1]["date"],
                df[df["label"] == 1]["close"],
                label="Label",
                color="red",
                marker="o",
            )  # ラベルをプロット
            plt.xlabel("Date")
            plt.ylabel("Close Price")
            plt.title("Close Price with Labels")
            plt.legend()  # 凡例を表示

            # x軸の目盛りを月単位に設定し、フォーマットを適用
            plt.gca().xaxis.set_major_locator(
                plt.matplotlib.dates.MonthLocator()
            )  # x軸の目盛りを月単位に設定
            plt.gca().xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter("%Y-%m")
            )  # x軸の目盛りのフォーマットを設定

            plt.xticks(rotation=45)  # x軸の目盛りを45度回転
            plt.grid(True)  # グリッドを表示
            plt.tight_layout()  # レイアウトを調整
            plt.show()  # プロットを表示
        except Exception as e:
            print(f"チャートプロットに失敗しました: {e}")


# 使用例
if __name__ == "__main__":
    data_path = "data/label/demo_labels.csv"
    manager = LabelDataManager(data_path)

    # データをロード
    df = manager.load_label_data()

    # チャートをプロット
    ChartPlotter.plot_data(df)
