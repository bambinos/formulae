from copy import deepcopy

import numpy as np
import pandas as pd

from packaging.version import Version
from scipy import sparse


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
    if Version(pd.__version__) < Version("2.1.0"):
        return pd.api.types.is_categorical_dtype(arr_or_dtype)
    else:
        if hasattr(arr_or_dtype, "dtype"):  # it's an array
            dtype = getattr(arr_or_dtype, "dtype")
        else:
            dtype = arr_or_dtype
        return isinstance(dtype, pd.CategoricalDtype)


def row_khatri_rao_sparse(X, groups, k):
    """Efficiently computes the 'row-wise Khatri-Rao' product for design matrices.


    Parameters
    ----------
    X : np.ndarray
        The model matrix for the random effects (e.g., intercepts and slopes) of shape (n * k)

    groups : np.ndarray
        The integer group indices for each observation (0 to k- 1).

    k : int
        Number of groups in ``groups``.

    Returns
    -------
    Z : scipy.sparse.csr_matrix
        The sparse design matrix of shape (n, p * k)
    """
    assert X.ndim == 2
    assert groups.ndim == 1
    assert len(groups) == X.shape[0]

    n, p = X.shape

    # Determine number of groups
    # k = groups.max() + 1

    # Construct the CSR internals directly
    # data: flattened X (row-wise, numpy's default)
    data = X.flatten()

    # indptr: Row offsets
    # Every row contains exactly p non-zero elements,
    # thus it goes from 0 to (n * p) in steps of size p
    # [0, p, 2 * p, ..., (n - 1) * p, n * p]
    indptr = np.arange(0, (n + 1) * p, p)

    # indices: Column indices for every value in 'data'
    # For each row there are 'p' consecutive values,
    # starting at a position that is a multiple of 'p'
    # For row i, group g, the columns are: [g * p, g * p + 1, ..., g * p + (p - 1)]
    # We use broadcasting to create this block for all rows at once
    # shape (n, p) -> flatten to (n * p,)
    indices_starts = groups * p
    indices = (indices_starts[:, np.newaxis] + np.arange(p)).flatten()

    # Create output patrix (CSR format)
    return sparse.csr_matrix((data, indices, indptr), shape=(n, k * p))
