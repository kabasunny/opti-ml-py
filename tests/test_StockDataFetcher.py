import sys
import os

# プロジェクトのルートディレクトリを sys.path に追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import pytest
from data.StockDataFetcher import StockDataFetcher


@pytest.fixture
def stock_data_fetcher():
    symbol = "7203.T"
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2023-12-31")
    return StockDataFetcher(symbol, start_date, end_date)


def test_fetch_data(stock_data_fetcher):
    stock_data = stock_data_fetcher.fetch_data()
    assert stock_data is not None
    assert not stock_data.data.empty


def test_data_columns(stock_data_fetcher):
    stock_data = stock_data_fetcher.fetch_data()
    expected_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert list(stock_data.data.columns) == expected_columns


def test_data_dates(stock_data_fetcher):
    stock_data = stock_data_fetcher.fetch_data()
    assert stock_data.data.index.min() == pd.Timestamp("2023-01-04")
    assert stock_data.data.index.max() == pd.Timestamp("2023-12-29")
