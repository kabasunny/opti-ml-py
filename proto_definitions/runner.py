import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from proto_conversion import convert_to_proto_response
from proto_definitions.print_ml_stock_response import print_ml_stock_response_summary

if __name__ == "__main__":
    # 使用例
    signals_csv_path = "data/stock_data/predictions/1570_2025-01-12.csv"
    daily_data_csv_path = "data/stock_data/formated_raw/1570_2025-01-12.csv"

    ml_stock_response = convert_to_proto_response(signals_csv_path, daily_data_csv_path)
    print_ml_stock_response_summary(ml_stock_response)
