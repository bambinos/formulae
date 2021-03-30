import numpy as np
import pandas as pd

# Stateful transformations.
# These transformations have memory about the state of parameters that are
# required to compute the transformation and are obtained as a subproduct of the
# data that is used to compute the transform.


class Center:
    def __init__(self):
        self.params_set = False
        self.mean = None

    def __call__(self, x):
        if not self.params_set:
            self.mean = np.mean(x)
            self.params_set = True
        return x - self.mean


class Scale:
    def __init__(self):
        self.params_set = False
        self.mean = None
        self.std = None

    def __call__(self, x):
        if not self.params_set:
            self.mean = np.mean(x)
            self.std = np.std(x)
            self.params_set = True
        return (x - self.mean) / self.std


# The following are just regular functions that are made available
# in the environment where the formula is evaluated.
def I(x):
    """Identity function. Returns its argument as it is.

    This allows to call Python code within the formula interface.
    This is an allias for ``{x}``, which does exactly the same, but in a more concise manner.

    Examples
    ----------

    >>> x + I(x**2)
    >>> x + {x**2}
    >>> {(x + y) / z}
    """
    return x


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
    if ref is not None:
        bool_ = x == ref
        if sum(bool_) == 0:
            raise ValueError(f"No value in 'x' is equal to 'ref' \"{ref}\"")
        value = np.atleast_2d(np.where(bool_, 1, 0)).T
        return value
    elif levels is not None:
        cat_type = pd.api.types.CategoricalDtype(categories=levels, ordered=True)
        x = x.astype(cat_type)
    elif not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
        categories = sorted(x.unique().tolist())
        cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
        x = x.astype(cat_type)
    return x


TRANSFORMS = {"I": I, "C": C}
STATEFUL_TRANSFORMS = {"center": Center, "scale": Scale, "standardize": Scale}
