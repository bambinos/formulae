from copy import deepcopy

import numpy as np
import pandas as pd


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple)) else [obj]


def flatten_list(nested_list):
    """Flatten a nested list"""
    nested_list = deepcopy(nested_list)
    while nested_list:
        sublist = nested_list.pop(0)
        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


def get_interaction_matrix(x, y):
    l = []

    if x.ndim == 1:
        x = x[:, np.newaxis]

    if y.ndim == 1:
        y = y[:, np.newaxis]

    for j1 in range(x.shape[1]):
        for j2 in range(y.shape[1]):
            l.append(x[:, j1] * y[:, j2])
    return np.column_stack(l)


def is_categorical_dtype(arr_or_dtype):
    """Check whether an array-like or dtype is of the pandas Categorical dtype."""
    # https://pandas.pydata.org/docs/whatsnew/v2.1.0.html#other-deprecations
    if pd.__version__ < "2.1.0":
        return pd.api.types.is_categorical_dtype(arr_or_dtype)
    else:
        if hasattr(arr_or_dtype, "dtype"):  # it's an array
            dtype = getattr(arr_or_dtype, "dtype")
        else:
            dtype = arr_or_dtype
        return isinstance(dtype, pd.CategoricalDtype)
