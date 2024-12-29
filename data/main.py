import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.JQuantsStockDataFetcher import JQuantsStockDataFetcher


def main(fetcher):
    raw_data = fetcher.fetch_data()
    standardized_data = fetcher.standardize_data(raw_data)
    if standardized_data.empty:
        print("No data found for the specified parameters.")
        return

    print("Daily data:")
    print(standardized_data.head())

    # 保存先ディレクトリの確認と作成
    save_dir = "data/test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # データをファイルに保存する例
    save_path = os.path.join(save_dir, "stock_data.csv")
    standardized_data.to_csv(save_path, index=False)

    # データ構造の確認
    print("\nデータの詳細情報 : standardized_data.info()")
    print(standardized_data.info())

    print("\nデータの基本統計量 : standardized_data.describe()")
    print(standardized_data.describe())

    print("\nデータのカラム名 : standardized_data.columns")
    print(standardized_data.columns)


# フラグを交互に切り替える関数
def toggle_use_jquants(flag):
    return not flag


if __name__ == "__main__":
    # 環境変数や設定ファイルを使用して実装クラスを切り替え
    use_jquants = True  # 初期値をTrueに設定
    max_iterations = 2  # 最大ループ回数を設定
    for _ in range(max_iterations):  # ここでは2回交互に実行します
        if use_jquants:
            print("\n★JQuantsStockDataFetcher★")
            fetcher = JQuantsStockDataFetcher(
                "7203", "2023-01-01", "2023-12-31"
            )  # 株式コードから".T"を削除
        else:
            print("\n★YahooFinanceStockDataFetcher★")
            fetcher = YahooFinanceStockDataFetcher(
                "7203", "2023-01-01", "2023-12-31"
            )  # 株式コードから".T"を削除
        main(fetcher)
        # フラグを交互に切り替え
        use_jquants = toggle_use_jquants(use_jquants)
