import os
import pandas as pd
from proto_definitions.ml_stock_service_pb2 import (
    MLStockResponse,
    MLSymbolData,
    MLDailyData,
    ModelPredictions,
)


def load_signals_csv(file_path):
    df = pd.read_csv(file_path)
    signal_dates = df[df["label"] == 1]["date"].tolist()
    return signal_dates


def load_daily_data_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def convert_to_proto_response(signals_csv_path, daily_data_csv_path):
    signal_dates = load_signals_csv(signals_csv_path)
    daily_data_df = load_daily_data_csv(daily_data_csv_path)

    symbol = str(daily_data_df["symbol"].iloc[0])  # Symbol should be a string

    daily_data_list = [
        MLDailyData(
            date=str(row["date"]),  # Ensure date is a string
            open=float(row["open"]),  # Ensure open is a float
            high=float(row["high"]),  # Ensure high is a float
            low=float(row["low"]),  # Ensure low is a float
            close=float(row["close"]),  # Ensure close is a float
            volume=int(row["volume"]),  # Ensure volume is an integer
        )
        for index, row in daily_data_df.iterrows()
    ]

    model_predictions = {}
    signals_df = pd.read_csv(
        signals_csv_path
    )  # 追加: シグナルCSVファイルを再度読み込み
    for model in [
        "LightGBM",
        "RandomForest",
        "XGBoost",
        "CatBoost",
        "AdaBoost",
        "SVM",
        "KNeighbors",
        "LogisticRegression",
    ]:
        prediction_dates = signals_df[signals_df[model] == 1]["date"].tolist()
        model_predictions[model] = ModelPredictions(
            prediction_dates=[str(date) for date in prediction_dates]
        )  # Ensure dates are strings

    symbol_data = MLSymbolData(
        symbol=symbol,
        daily_data=daily_data_list,
        signals=[str(signal) for signal in signal_dates],  # Ensure signals are strings
        model_predictions=model_predictions,  # Add model predictions
    )

    ml_stock_response = MLStockResponse(symbol_data=[symbol_data])

    return ml_stock_response


def save_proto_response_to_file(proto_response, save_directory, csv_file_path):
    # Extract the file name without extension from the CSV file path
    base_filename = os.path.basename(csv_file_path)
    filename_without_extension = os.path.splitext(base_filename)[0]

    # Create the save file path
    save_file_path = os.path.join(save_directory, f"{filename_without_extension}.bin")

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
