import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from proto_conversion import (
    convert_to_proto_response,
    save_proto_response_to_file,
    load_proto_response_from_file,
)
from proto_definitions.print_ml_stock_response import print_ml_stock_response_summary

if __name__ == "__main__":
    # 使用例
    signals_csv_path = "data/stock_data/predictions/7203_2025-01-15.csv"
    daily_data_csv_path = "data/stock_data/formated_raw/7203_2025-01-15.csv"
    save_directory = "../go-optimal-stop/data/ml_stock_response"

    # Convert and save proto response
    ml_stock_response = convert_to_proto_response(signals_csv_path, daily_data_csv_path)
    save_file_path = save_proto_response_to_file(
        ml_stock_response, save_directory, signals_csv_path
    )

    # Load and print proto response
    loaded_proto_response = load_proto_response_from_file(save_file_path)
    print_ml_stock_response_summary(loaded_proto_response)


# MLStockResponse (summary):
# Symbol: 1570
#   Daily Data:
#     Date: 2012-04-09, Open: 4351.0, Close: 4351.0
#     Date: 2012-04-10, Open: 4345.35986328125, Close: 4345.35986328125
#     ...
#     Date: 2025-01-09, Open: 28050.0, Close: 27740.0
#     Date: 2025-01-10, Open: 27295.0, Close: 27175.0
#   Signals: ['2014-05-19', '2014-05-20'] ... ['2024-12-19', '2024-12-20']
#   Model Predictions:
#     AdaBoost: ['2014-10-16', '2014-10-20'] ... ['2024-09-12', '2024-09-18']
#     KNeighbors: ['2014-10-22', '2014-10-24'] ... ['2024-08-02', '2024-09-12']
#     XGBoost: ['2014-10-16', '2014-10-20'] ... ['2024-09-18', '2024-09-19']
#     SVM: [] ... []
#     LightGBM: ['2014-10-20', '2014-10-21'] ... ['2024-09-13', '2024-09-18']
#     RandomForest: ['2014-10-16', '2014-10-24'] ... ['2024-09-13', '2024-09-18']
#     CatBoost: ['2014-10-16', '2014-10-20'] ... ['2024-09-18', '2024-09-19']
#     LogisticRegression: ['2015-09-03', '2015-09-08'] ... ['2024-08-06', '2024-08-09']
