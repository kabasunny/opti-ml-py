import yfinance as yf
import pandas as pd
from data.StockDataFetcherABC import StockDataFetcherABC
from decorators.ArgsChecker import ArgsChecker  # デコレータクラスをインポート


class YahooFinanceStockDataFetcher(StockDataFetcherABC):
    @ArgsChecker((None, (str, pd.Timestamp), (str, pd.Timestamp)), None)
    def __init__(self):
        pass

    @ArgsChecker((None, str), pd.DataFrame)
    def fetch_data(self, symbol) -> pd.DataFrame:
        self.symbol = symbol + ".T"
        # 全期間のデータを取得
        data = yf.Ticker(self.symbol).history(period="max")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data

    @ArgsChecker((None, pd.DataFrame, str), pd.DataFrame)
    def format_data(self, data: pd.DataFrame, symbol) -> pd.DataFrame:
        data = data.reset_index()
        data["symbol"] = symbol
        data = data.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        # `date`を文字列に変換
        data["date"] = data["date"].astype(str)
        # print(data)
        # タイムゾーンの情報を削除して比較可能にする
        data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
        # print(data)
        return data[["date", "symbol", "open", "high", "low", "close", "volume"]]


# Symbol of current data: 1570

# タイムゾーン有
#                            date          open  ...  Capital Gains  symbol
# 0     2012-04-09 00:00:00+09:00   4351.000000  ...            0.0    1570
# 1     2012-04-10 00:00:00+09:00   4345.359863  ...            0.0    1570
# 2     2012-04-11 00:00:00+09:00   4254.160156  ...            0.0    1570
# 3     2012-04-12 00:00:00+09:00   2157.500000  ...            0.0    1570
# 4     2012-04-13 00:00:00+09:00   2230.000000  ...            0.0    1570
# ...                         ...           ...  ...            ...     ...
# 3136  2025-01-06 00:00:00+09:00  28070.000000  ...            0.0    1570
# 3137  2025-01-07 00:00:00+09:00  27830.000000  ...            0.0    1570
# 3138  2025-01-08 00:00:00+09:00  28000.000000  ...            0.0    1570
# 3139  2025-01-09 00:00:00+09:00  28050.000000  ...            0.0    1570
# 3140  2025-01-10 00:00:00+09:00  27295.000000  ...            0.0    1570
# [3141 rows x 10 columns]


# タイムゾーン無し
#            date          open          high  ...  Stock Splits  Capital Gains  symbol
# 0    2012-04-09   4351.000000   4351.000000  ...           0.0            0.0    1570
# 1    2012-04-10   4345.359863   4345.359863  ...           0.0            0.0    1570
# 2    2012-04-11   4254.160156   4254.160156  ...           0.0            0.0    1570
# 3    2012-04-12   2157.500000   2182.500000  ...           0.0            0.0    1570
# 4    2012-04-13   2230.000000   2250.000000  ...           0.0            0.0    1570
# ...         ...           ...           ...  ...           ...            ...     ...
# 3136 2025-01-06  28070.000000  28265.000000  ...           0.0            0.0    1570
# 3140 2025-01-10  27295.000000  27500.000000  ...           0.0            0.0    1570
# [3141 rows x 10 columns]
