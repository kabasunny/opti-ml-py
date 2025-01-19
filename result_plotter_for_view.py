import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_types import model_types  # 別ファイルで定義
import os

symbol = "2914"
date = "2025-01-19"

# データの読み込み
stock_data_path = f"data/stock_data/formated_raw/{date}/{symbol}.csv"
predictions_data_path = f"data/stock_data/real_predictions/{date}/{symbol}.csv"
labels_data_path = f"data/stock_data/labeled/{date}/{symbol}.csv"

stock_data = pd.read_csv(stock_data_path)
predictions_data = pd.read_csv(predictions_data_path)
labels_data = pd.read_csv(labels_data_path)

# 日付列をdatetime型に変換
stock_data["date"] = pd.to_datetime(stock_data["date"])
predictions_data["date"] = pd.to_datetime(predictions_data["date"])
labels_data["date"] = pd.to_datetime(labels_data["date"])

# プロット用のデータフレームを作成
plot_data = pd.merge(stock_data, predictions_data, on=["date", "symbol"])
plot_data = pd.merge(
    plot_data, labels_data, on=["date", "symbol"], suffixes=("", "_label")
)

# モデルごとにプロットを作成
for model in model_types:
    if model in plot_data.columns:
        plt.figure(figsize=(14, 7))

        # 株価のチャートを作成
        sns.lineplot(data=plot_data, x="date", y="close", label="Stock Price")

        # モデルの予測をプロット
        model_predictions = plot_data[plot_data[model] == 1]
        num_predictions = len(model_predictions)
        plt.scatter(
            model_predictions["date"],
            model_predictions["close"],
            label=f"{model} Prediction (Count: {num_predictions})",
            s=50,
            marker="^",  # 三角に設定
            color="#00FF00",  # 蛍光色の緑に設定
        )

        # 正解ラベルをプロット
        true_labels = plot_data[plot_data["label"] == 1]
        num_true_labels = len(true_labels)
        plt.scatter(
            true_labels["date"],
            true_labels["close"],
            label=f"True Label (Count: {num_true_labels})",
            s=50,
            marker="o",  # 丸に設定
            color="red",  # 赤に設定
        )

        # 一致ポイントをプロット
        matched_points = plot_data[(plot_data[model] == 1) & (plot_data["label"] == 1)]
        num_matched_points = len(matched_points)
        plt.scatter(
            matched_points["date"],
            matched_points["close"],
            label=f"Matched Points (Count: {num_matched_points})",
            s=50,
            marker="s",  # 四角に設定
            color="#FF00FF",  # 蛍光色の紫に設定
        )

        # プロットの装飾
        plt.title(f"{model} Predictions on Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()

        # プロットを表示
        plt.show()
        plt.close()
