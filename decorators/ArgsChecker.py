# opti-ml-py\decorators\ArgsChecker.py
import pandas as pd
from functools import wraps


class ArgsChecker:
    def __init__(self, arg_types=(), return_type=None):
        self.arg_types = arg_types
        self.return_type = return_type

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 位置引数の型チェック
            for i, (arg, expected_type) in enumerate(zip(args, self.arg_types)):
                if expected_type is None:
                    continue
                if isinstance(expected_type, tuple):
                    if not any(isinstance(arg, t) for t in expected_type):
                        raise TypeError(
                            f"Argument {i} should be of type one of {[t.__name__ for t in expected_type]}, "
                            f"but got {type(arg).__name__}"
                        )
                elif not isinstance(arg, expected_type):
                    raise TypeError(
                        f"Argument {i} is of type {type(arg).__name__} but should be {expected_type.__name__}"
                    )

            result = func(*args, **kwargs)

            if self.return_type is not None:
                if not isinstance(result, self.return_type):
                    raise TypeError(
                        f"Return value should be of type {self.return_type.__name__}, but got {type(result).__name__}"
                    )

            return result

        return wrapper
