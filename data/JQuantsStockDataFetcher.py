import os
import requests
import pandas as pd
from dotenv import load_dotenv
from data.StockDataFetcherBase import StockDataFetcherBase
import time
from data.decorators import type_check  # デコレータをインポート


class JQuantsStockDataFetcher(StockDataFetcherBase):
    @type_check((None, str, (str, pd.Timestamp), (str, pd.Timestamp)), None)
    def __init__(self, symbol, start_date, end_date):
        self.symbol = str(symbol)  # symbol を文字列にキャスト
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

    @type_check((None,), str)
    def get_refresh_token(self):
        email = os.getenv("JQUANTS_EMAIL")
        password = os.getenv("JQUANTS_PASSWORD")
        url = "https://api.jquants.com/v1/token/auth_user"
        headers = {"Content-Type": "application/json"}
        data = {"mailaddress": email, "password": password}

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("refreshToken")

    @type_check((None, str), str)
    def get_id_token(self, refresh_token):
        url = f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={refresh_token}"
        response = requests.post(url)
        response.raise_for_status()
        return response.json().get("idToken")

    @type_check((None,), pd.DataFrame)
    def fetch_data(self) -> pd.DataFrame:
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
                response.raise_for_status()
                result = response.json()
                data.extend(result.get("daily_quotes", []))
                if "pagination_key" not in result:
                    break
                params["pagination_key"] = result["pagination_key"]
            except requests.exceptions.HTTPError as e:
                print(f"Error fetching data for symbol {self.symbol}: {e}")
                break

            time.sleep(1)  # 1秒の待機

        df = pd.DataFrame(data)
        return df

    @type_check((None, pd.DataFrame), pd.DataFrame)
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
        data["symbol"] = data["Code"].str[:4]
        data["volume"] = data["volume"].astype(int)
        data["date"] = data["date"].astype(str)
        return data[["date", "symbol", "open", "high", "low", "close", "volume"]]
