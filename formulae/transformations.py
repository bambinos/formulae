import numpy as np
import pandas as pd


def I(x):
    """Returns its argument as it is.

    This allows to call Python code  within the formula interface.

    Examples
    ----------

    >>> x + I(x**2)
    >>> x + {x**2}
    >>> {(x + y) / z}
    """
    return x


def center(x):
    """Centers a numerical variable"""
    return x - np.mean(x)


def scale(x):
    """Standardize a numerical variable"""
    return (x - np.mean(x)) / np.std(x)


def C(x, ref=None, levels=None):
    """Make a variable categorical

    This is an internal function only accesible through the formula interface.
    `ref` takes precedence over `levels`

    Parameters
    ----------

    x: pd.Series or 1D np.array
        The object containing the variable to be converted to categorical.
    ref: str, numeric or None
        The reference level. This is used when the desired output is a 0-1 variable.
        The reference level is 1 and the rest are 0. Defaults to None which means this
        feature is disabled and the variable is categorized using a dummy encoding according
        to the levels specified in `levels` or the order the levels appear in the variable.
    levels: list or None
        A list describing the desired order for the categorical variable. Defaults to None
        which means `ref` is used if not None.

    Returns
    ----------
    x: pd.Series
        An ordered categorical series.
    """

    if ref is not None and levels is not None:
        raise ValueError("At least one of 'ref' or 'levels' must be None.")
    elif ref is not None:
        bool_ = x == ref
        if sum(bool_) == 0:
            raise ValueError(f"No value in 'x' is equal to 'ref' \"{ref}\"")
        value = np.atleast_2d(np.where(bool_, 1, 0)).T
        return value
    elif levels is not None:
        cat_type = pd.api.types.CategoricalDtype(categories=levels, ordered=True)
        x = x.astype(cat_type)
    elif not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
        cat_type = pd.api.types.CategoricalDtype(categories=x.unique().tolist(), ordered=True)
        x = x.astype(cat_type)
    return x


TRANSFORMATIONS = {"I": I, "center": center, "scale": scale, "standardize": scale, "C": C}
