import time
import pandas as pd
from functools import wraps


# 型チェックデコレータの定義
def type_check(arg_types, return_type=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i, (arg, expected_type) in enumerate(zip(args, arg_types)):
                if not isinstance(arg, expected_type):
                    raise TypeError(
                        f"Argument {i} should be of type {expected_type.__name__}"
                    )
            result = func(*args, **kwargs)
            if return_type and not isinstance(result, return_type):
                raise TypeError(
                    f"Return value should be of type {return_type.__name__}"
                )
            return result

        return wrapper

    return decorator


# 手動型チェックのクラス定義
class StockDataManual:
    def __init__(self, symbol, data):
        if not isinstance(symbol, str):
            raise TypeError("Symbol should be a string")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame")
        self.symbol = symbol
        self.data = data

    def save_to_csv(self, filepath):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("Data should be a pandas DataFrame")
        if not isinstance(filepath, str):
            raise TypeError("File path should be a string")
        self.data.to_csv(filepath, index=True)


# デコレータ型チェックのクラス定義
class StockDataDecorator:
    @type_check((str, pd.DataFrame))
    def __init__(self, symbol, data):
        print(f"symbol: {symbol}, type: {type(symbol)}")
        print(f"data: {data}, type: {type(data)}")
        self.symbol = symbol
        self.data = data

    @type_check((str,))
    def save_to_csv(self, filepath):
        self.data.to_csv(filepath, index=True)


# ベンチマーク関数の定義
def benchmark(func):
    start_time = time.time()
    func()
    end_time = time.time()
    return end_time - start_time


# テスト用データ
data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

# 手動型チェックのベンチマーク
manual_time = benchmark(
    lambda: StockDataManual("symbol", data).save_to_csv("manual.csv")
)
# デコレータ型チェックのベンチマーク
decorator_time = benchmark(
    lambda: StockDataDecorator("symbol", data).save_to_csv("decorator.csv")
)

print(f"Manual type check time: {manual_time}")
print(f"Decorator type check time: {decorator_time}")
