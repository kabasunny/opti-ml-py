import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from proto_conversion.print_ml_stock_response import print_ml_stock_response_summary
from data.DataManager import DataManager
from datetime import datetime
from proto_conversion.ProtoConvertPipeline import ProtoConvertPipeline
from proto_conversion.ProtoSaverLoader import ProtoSaverLoader
from symbols import symbols  # 別ファイルで定義
from model_types import model_types  # 別ファイルで定義

if __name__ == "__main__":
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    base_data_path = "data/stock_data"
    file_ext = "csv"  # "parquet"

    raw_data_manager = DataManager(
        current_date_str, base_data_path, "formated_raw", file_ext
    )
    real_pred_data_manager = DataManager(
        current_date_str, base_data_path, "Real_predictions", file_ext
    )
    file_path = "../go-optimal-stop/data/ml_stock_response/latest_response.bin"

    # ProtoSaverLoaderの初期化
    proto_saver_loader = ProtoSaverLoader(file_path)

    # ProtoConvertPipelineの初期化と実行
    ProtoConvertPipeline(raw_data_manager, real_pred_data_manager, proto_saver_loader, model_types).run(symbols) # リストを受けるため他のパイプラインと異なる
    print("Proto conversion and saving completed.")


    # 保存したプロトコルバッファーの読み込み
    loaded_proto_response = proto_saver_loader.load_proto_response_from_file()
    print_ml_stock_response_summary(loaded_proto_response)
