import os
import pandas as pd
from proto_definitions.ml_stock_service_pb2 import (
    MLStockResponse,
    MLSymbolData,
    MLDailyData,
    ModelPredictions,
)
from datetime import datetime


def convert_to_proto_response(raw_data_manager, predictions_data_manager, symbols):
    responses = []
    for symbol in symbols:
        # raw_data_manager からデータを読み込む
        raw_data_df = raw_data_manager.load_data(symbol)
        daily_data_list = [
            MLDailyData(
                date=str(row["date"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            for index, row in raw_data_df.iterrows()
        ]

        # predictions_data_manager からデータを読み込む
        predictions_df = predictions_data_manager.load_data(symbol)
        signal_dates = predictions_df[predictions_df["label"] == 1]["date"].tolist()

        model_predictions = {}
        for model in [
            "LightGBM",
            "RandomForest",
            "XGBoost",
            "CatBoost",
            "AdaBoost",
            "DecisionTree",
            "GradientBoosting",
            "ExtraTrees",
            "Bagging",
            "Voting",
            "Stacking",
        ]:
            prediction_dates = predictions_df[predictions_df[model] == 1][
                "date"
            ].tolist()
            model_predictions[model] = ModelPredictions(
                prediction_dates=[str(date) for date in prediction_dates]
            )

        symbol_data = MLSymbolData(
            symbol=symbol,
            daily_data=daily_data_list,
            signals=[str(signal) for signal in signal_dates],
            model_predictions=model_predictions,
        )

        responses.append(symbol_data)  # MLSymbolData を直接追加

    # MLSymbolData のリストを含む結合レスポンスを作成
    combined_response = MLStockResponse(symbol_data=responses)
    return combined_response


def save_proto_response_to_file(proto_response, save_directory):
    # Get the current date and time to use as the file name
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_file_path = os.path.join(save_directory, f"{current_date}.bin")

    # Save the proto response to a binary file
    with open(save_file_path, "wb") as f:
        f.write(proto_response.SerializeToString())

    return save_file_path


def load_proto_response_from_file(file_path):
    # Load the proto response from a binary file
    with open(file_path, "rb") as f:
        proto_response = MLStockResponse()
        proto_response.ParseFromString(f.read())
    return proto_response
