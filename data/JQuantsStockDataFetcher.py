import os
import requests
import pandas as pd
from dotenv import load_dotenv
from data.StockDataFetcherBase import StockDataFetcherBase
import time


class JQuantsStockDataFetcher(StockDataFetcherBase):
    def __init__(self, symbol, start_date, end_date):
        if not isinstance(symbol, str):
            raise TypeError("Symbol should be a string")
        if not isinstance(start_date, (str, pd.Timestamp)):
            raise TypeError("Start date should be a string or pandas Timestamp")
        if not isinstance(end_date, (str, pd.Timestamp)):
            raise TypeError("End date should be a string or pandas Timestamp")

        self.symbol = symbol
        self.start_date = (
            pd.Timestamp(start_date) if isinstance(start_date, str) else start_date
        )
        self.end_date = (
            pd.Timestamp(end_date) if isinstance(end_date, str) else end_date
        )

        # 環境変数を読み込み
        load_dotenv()
        self.refresh_token = self.get_refresh_token()
        self.id_token = self.get_id_token(self.refresh_token)

    def get_refresh_token(self):
        """リフレッシュトークンを取得する"""
        email = os.getenv("JQUANTS_EMAIL")
        password = os.getenv("JQUANTS_PASSWORD")
        url = "https://api.jquants.com/v1/token/auth_user"
        headers = {"Content-Type": "application/json"}
        data = {"mailaddress": email, "password": password}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # ステータスコードが200番台以外の場合に例外を投げる
        return response.json().get("refreshToken")

    def get_id_token(self, refresh_token):
        """リフレッシュトークンを使ってIDトークンを取得する"""
        url = f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={refresh_token}"
        response = requests.post(url)
        response.raise_for_status()  # ステータスコードが200番台以外の場合に例外を投げる
        return response.json().get("idToken")

    def fetch_data(self) -> pd.DataFrame:
        """日次株価データを取得する"""
        url = "https://api.jquants.com/v1/prices/daily_quotes"
        headers = {"Authorization": f"Bearer {self.id_token}"}
        params = {
            "code": self.symbol,
            "from": self.start_date.strftime("%Y-%m-%d"),
            "to": self.end_date.strftime("%Y-%m-%d"),
        }

        data = []
        while True:
            response = requests.get(url, headers=headers, params=params)
            try:
                response.raise_for_status()  # ステータスコードが200番台以外の場合に例外を投げる
                result = response.json()
                data.extend(result.get("daily_quotes", []))
                if "pagination_key" not in result:
                    break
                params["pagination_key"] = result["pagination_key"]
            except requests.exceptions.HTTPError as e:
                print(f"Error fetching data for symbol {self.symbol}: {e}")
                break

            # レートリミット対応のための待機時間
            time.sleep(1)  # 1秒の待機

        df = pd.DataFrame(data)
        return df

    def standardize_data(self, data: pd.DataFrame) -> pd.DataFrame:
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
        # シンボルの先頭4桁のみを返す
        data["symbol"] = data["Code"].str[:4]
        data["volume"] = data["volume"].astype(int)  # 小数を整数に変換
        # `date`を文字列に変換
        data["date"] = data["date"].astype(str)
        return data[["date", "symbol", "open", "high", "low", "close", "volume"]]
