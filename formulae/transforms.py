import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype

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


def C(x, reference=None, levels=None):
    """Make a variable categorical or manipulate the order of its levels.

    This is an internal function only accesible through the formula interface.

    Parameters
    ----------

    x: pd.Series
        The object containing the variable to be converted to categorical.
    reference: str, numeric or None
        The reference level. This is used when the goal is only to only change the level taken
        as reference but not the order of the others. Defaults to ``None`` which means this
        feature is disabled and the variable is categorized according to the levels specified in
        ``levels`` or the order of the levels after calling ``sorted()``.
    levels: list or None
        A list describing the desired order for the categorical variable. Defaults to ``None``
        which means either ``reference`` is used or the order of the levels after calling
        ``sorted()``.

    Returns
    ----------
    x: pd.Series
        An ordered categorical series.
    """

    if reference is not None and levels is not None:
        raise ValueError("At least one of 'reference' or 'levels' must be None.")

    if reference is not None:
        # If the variable has categories, use their order.
        if hasattr(x.dtype, "categories"):
            categories = list(x.dtype.categories)
        # If the variable does not have categories use `sorted()`.
        else:
            categories = sorted(x.unique().tolist())
        # Send reference to the first place
        categories.insert(0, categories.pop(categories.index(reference)))
    elif levels is not None:
        categories = levels
    elif not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
        categories = sorted(x.unique().tolist())

    # Create type and use it in the variable
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
    # successes and trials are pd.Series
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
