import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.YahooFinanceStockDataFetcher import YahooFinanceStockDataFetcher
from data.JQuantsStockDataFetcher import JQuantsStockDataFetcher
from data.DataSaver import DataSaver  # DataSaver クラスのインポート
from data.DataPipeline import DataPipeline  # DataPipeline クラスのインポート


def runner(fetcher):
    save_dir = "data/test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "stock_data.csv")

    saver = DataSaver()  # DataSaver クラスのインスタンスを作成
    pipeline = DataPipeline(fetcher, saver)  # DataPipeline クラスのインスタンスを作成
    pipeline.run(save_path)  # データパイプラインを実行


# フラグを交互に切り替える関数
def toggle(flag):
    return not flag


if __name__ == "__main__":
    use_jquants = True  # 初期値をTrueに設定
    max_iterations = 2  # 最大ループ回数を設定
    for _ in range(max_iterations):  # ここでは2回交互に実行します
        if use_jquants:
            print("\n★JQuantsStockDataFetcher★")
            fetcher = JQuantsStockDataFetcher("7203", "2023-01-01", "2023-12-31")
        else:
            print("\n★YahooFinanceStockDataFetcher★")
            fetcher = YahooFinanceStockDataFetcher("7203", "2023-01-01", "2023-12-31")
        runner(fetcher)  # メソッド名を runner に変更
        use_jquants = toggle(use_jquants)  # フラグを交互に切り替え
