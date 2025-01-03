# opti-ml-py\decorators\KwargsChecker.py
import pandas as pd
from functools import wraps


class KwargsChecker:
    def __init__(self, kwarg_types={}, return_type=None):
        self.kwarg_types = kwarg_types
        self.return_type = return_type

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # キーワード引数の型チェック
            for key, expected_type in self.kwarg_types.items():
                if key in kwargs:
                    arg = kwargs[key]
                    if isinstance(expected_type, tuple):
                        if not any(isinstance(arg, t) for t in expected_type):
                            raise TypeError(
                                f"Keyword argument '{key}' should be of type one of {[t.__name__ for t in expected_type]}, "
                                f"but got {type(arg).__name__}"
                            )
                    elif not isinstance(arg, expected_type):
                        raise TypeError(
                            f"Keyword argument '{key}' is of type {type(arg).__name__} but should be {expected_type.__name__}"
                        )

            result = func(*args, **kwargs)

            if self.return_type is not None:
                if not isinstance(result, self.return_type):
                    raise TypeError(
                        f"Return value should be of type {self.return_type.__name__}, but got {type(result).__name__}"
                    )

            return result

        return wrapper
