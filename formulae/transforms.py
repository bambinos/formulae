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


def binary(x, success=None):
    """Make a variable binary

    Parameters
    ----------
    x: pd.Series
        The object containing the variable to be converted to binary.
    success: str, numeric or None
        The success level. When the variable is equal to this level, the binary variable is 1.
        All the rest are 0. Defaults to ``None`` which means formulae is going to sort all the
        values in the variable and pick the first one as success.

    Returns
    -------
    x: np.array
        A 0-1 numpy array with shape ``(n, 1)`` where ``n`` is the number of observations.
    """
    if success is None:
        categories = sorted(x.unique().tolist())
        success = categories[0]
    booleans = x == success
    if not sum(booleans):
        raise ValueError(f"No value in 'x' is equal to \"{success}\"")
    return np.where(booleans, 1, 0)[:, np.newaxis]


class Prop:
    def __init__(self, successes, trials):
        if not (np.mod(successes, 1) == 0).all():
            raise ValueError("'successes' must be a collection of integer numbers")

        if not (np.mod(trials, 1) == 0).all():
            raise ValueError("'trials' must be a collection of integer numbers")

        if not (np.less_equal(successes, trials)).all():
            raise ValueError("'successes' cannot be greater than 'trials'")

        self.successes = successes
        self.trials = trials

    def eval(self):
        return np.vstack([self.successes, self.trials]).T


def prop(successes, trials):
    # Successes and trials are pd.Series
    successes = successes.values
    trials = trials.values
    return Prop(successes, trials)


class Offset:
    def __init__(self, x):
        if not is_numeric_dtype(x):
            raise ValueError("offset() can only be used with numeric variables.")
        self.x = x.values

    def eval(self):
        return self.x.flatten()[:, np.newaxis]


def offset(x):
    return Offset(x)


TRANSFORMS = {"I": I, "C": C, "binary": binary, "prop": prop, "offset": offset}
STATEFUL_TRANSFORMS = {"center": Center, "scale": Scale, "standardize": Scale}
