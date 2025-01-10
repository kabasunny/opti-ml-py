import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
stock_data_path = "data/stock_data/formated_raw/5803_2025-01-08.parquet"
predictions_data_path = "data/stock_data/predictions/5803_2025-01-08.parquet"

stock_data = pd.read_parquet(stock_data_path)
predictions_data = pd.read_parquet(predictions_data_path)

# 日付列をdatetime型に変換
stock_data["date"] = pd.to_datetime(stock_data["date"])
predictions_data["date"] = pd.to_datetime(predictions_data["date"])

# プロット用のデータフレームを作成
plot_data = pd.merge(stock_data, predictions_data, on=["date", "symbol"])

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
]

for model in models:
    plt.figure(figsize=(14, 7))

    # 株価のチャートを作成
    sns.lineplot(data=plot_data, x="date", y="close", label="Stock Price")

    # モデルの予測をプロット
    model_predictions = plot_data[plot_data[model] == 1]
    plt.scatter(
        model_predictions["date"],
        model_predictions["close"],
        label=f"{model} Prediction",
        s=50,  # サイズを半分に設定
        marker="o",
        color="red",  # 赤色に設定
    )

    # プロットの装飾
    plt.title(f"Stock Price and {model} Predictions")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

    # プロットを表示
    plt.show()
