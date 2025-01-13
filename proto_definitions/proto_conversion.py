import pandas as pd
from proto_definitions.ml_stock_service_pb2 import (
    MLStockResponse,
    MLSymbolData,
    MLDailyData,
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

    symbol_data = MLSymbolData(
        symbol=symbol,
        daily_data=daily_data_list,
        signals=[str(signal) for signal in signal_dates],  # Ensure signals are strings
    )

    ml_stock_response = MLStockResponse(symbol_data=[symbol_data])

    return ml_stock_response


# MLStockResponse (summary):
# Symbol: 1570
#   Daily Data:
#     Date: 2012-04-09, Open: 4351.0, Close: 4351.0
#     Date: 2012-04-10, Open: 4345.35986328125, Close: 4345.35986328125
#     ...
#     Date: 2025-01-09, Open: 28050.0, Close: 27740.0
#     Date: 2025-01-10, Open: 27295.0, Close: 27175.0
#   Signals: ['2014-05-19', '2014-05-20'] ... ['2024-12-19', '2024-12-20']
