import numpy as np
import pandas as pd


def I(x):
    return x


def center(x):
    return x - np.mean(x)


def scale(x):
    return (x - np.mean(x)) / np.std(x)


def C(x, ref=None, levels=None):

    if ref is not None and levels is not None:
        raise ValueError("At least one of 'ref' or 'levels' must be None.")
    if ref is not None:
        value = np.atleast_2d(np.where(x == ref, 1, 0)).T
        return {"value": value, "reference": ref}
    elif levels is not None:
        cat_type = pd.api.types.CategoricalDtype(categories=levels, ordered=True)
        x = x.astype(cat_type)

    elif not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
        cat_type = pd.api.types.CategoricalDtype(categories=x.unique().tolist(), ordered=True)
        x = x.astype(cat_type)
    return x


TRANSFORMATIONS = {"I": I, "center": center, "scale": scale, "standardize": scale, "C": C}
