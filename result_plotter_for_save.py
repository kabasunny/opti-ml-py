import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# データの読み込み
stock_data_path = "data/stock_data/formated_raw/7203_2025-01-15.csv"
predictions_data_path = "data/stock_data/predictions/7203_2025-01-15.csv"
labels_data_path = "data/stock_data/labeled/7203_2025-01-15.csv"
save_dir = "data/stock_data/png_image"

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


# ディレクトリが存在しない場合は作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# モデルごとにプロットを作成
models = [
    "LightGBM",
    "RandomForest",
    "XGBoost",
    "CatBoost",
    "AdaBoost",
    "SVM",
    "KNeighbors",
    "LogisticRegression",
    "DecisionTree",
    "GradientBoosting",
    "NaiveBayes",
    "RidgeRegression",
    "ExtraTrees",
    "Bagging",
    "Voting",
    "Stacking",
    "PassiveAggressive",
    "Perceptron",
    "SGD",
]

for model in models:
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

        # プロットを保存
        save_path = os.path.join(save_dir, f"{model}_predictions.png")
        plt.savefig(save_path)
        plt.close()

print(f"プロット画像が '{save_dir}' に保存されました。")
