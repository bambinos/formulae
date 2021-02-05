import numpy as np
import pandas as pd

def I(x):
    return x

def center(x):
    return x - np.mean(x)

def scale(x):
    return (x - np.mean(x)) / np.std(x)

def C(x, levels=None):
    if levels is not None:
        cat_type = pd.api.types.CategoricalDtype(categories=levels, ordered=True)
        x = x.astype(cat_type)
    elif not hasattr(x, "ordered") or not x.ordered:
        cat_type = pd.api.types.CategoricalDtype(categories=x.unique().tolist(), ordered=True)
        x = x.astype(cat_type)
    return x

TRANSFORMATIONS = {
    "I": I,
    "center": center,
    "scale":scale,
    "standardize": scale,
    "C": C
}