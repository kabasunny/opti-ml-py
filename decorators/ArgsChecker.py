# opti-ml-py\decorators\ArgsChecker.py
import pandas as pd
from functools import wraps
from typing import get_origin, get_args, Any, Tuple


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
                origin = get_origin(expected_type)
                if origin:
                    if not isinstance(arg, origin):
                        raise TypeError(
                            f"Argument {i} should be of type {origin.__name__}, "
                            f"but got {type(arg).__name__}"
                        )
                    for arg_item, expected_arg_type in zip(
                        arg, get_args(expected_type)
                    ):
                        if not isinstance(arg_item, expected_arg_type):
                            raise TypeError(
                                f"Elements of argument {i} should be of type {expected_arg_type.__name__}, "
                                f"but got {type(arg_item).__name__}"
                            )
                else:
                    if not isinstance(arg, expected_type):
                        raise TypeError(
                            f"Argument {i} is of type {type(arg).__name__} but should be {expected_type.__name__}"
                        )

            result = func(*args, **kwargs)

            if self.return_type is not None:
                origin = get_origin(self.return_type)
                if origin:
                    if not isinstance(result, origin):
                        raise TypeError(
                            f"Return value should be of type {origin.__name__}, "
                            f"but got {type(result).__name__}"
                        )
                    result_args = get_args(self.return_type)
                    for result_item, expected_return_type in zip(result, result_args):
                        if not isinstance(result_item, expected_return_type):
                            raise TypeError(
                                f"Elements of return value should be of type {expected_return_type}, "
                                f"but got {type(result_item).__name__}"
                            )
                elif isinstance(self.return_type, tuple):
                    for result_item, expected_return_type in zip(
                        result, self.return_type
                    ):
                        if not isinstance(result_item, expected_return_type):
                            raise TypeError(
                                f"Elements of return value should be of type {expected_return_type.__name__}, "
                                f"but got {type(result_item).__name__}"
                            )
                elif not isinstance(result, self.return_type):
                    raise TypeError(
                        f"Return value should be of type {self.return_type.__name__}, but got {type(result).__name__}"
                    )

            return result

        return wrapper
