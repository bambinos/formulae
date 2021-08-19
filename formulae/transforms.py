import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy.interpolate import splev

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
        ``sorted()``

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


class Proportion:
    """Representation of a proportion term.

    Parameters
    ----------
    successes: ndarray
        1D array containing data with ``int`` type.
    trials: ndarray
        1D array containing data with ``int`` type. Its values must be equal or larger than the
        values in ``successes``
    trials_type: str
        Indicates whether ``trials`` is a constant value or not. It can be either ``"constant"``
        or ``"variable"``.
    """

    def __init__(self, successes, trials, trials_type):
        if not (np.mod(successes, 1) == 0).all():
            raise ValueError("'successes' must be a collection of integer numbers")

        if not (np.mod(trials, 1) == 0).all():
            raise ValueError("'trials' must be a collection of integer numbers")

        if not (np.less_equal(successes, trials)).all():
            raise ValueError("'successes' cannot be greater than 'trials'")

        self.successes = successes
        self.trials = trials
        self.trials_type = trials_type

    def eval(self):
        return np.vstack([self.successes, self.trials]).T


def proportion(successes, trials):
    """Create a term that represents the proportion ``successes/trials``.

    This function is actually a wrapper of class ``Proportion`` that checks its arguments.

    Parameters
    ----------
    successes: pd.Series
        The number of successes for each observation unit.
    trials: pd.Series or int
        The number of trials for each observation unit. If ``int``, this function internally
        generates an array of the same length than ``successes``.
    """
    # If this function does not receive a pd.Series, it means the user didn't pass a name in the
    # formula interface

    if not isinstance(successes, pd.Series):
        raise ValueError("'successes' must be a variable name.")
    successes = successes.values

    if isinstance(trials, pd.Series):
        trials = trials.values
        trials_type = "variable"
    elif isinstance(trials, int):
        trials = np.ones(len(successes), dtype=int) * trials
        trials_type = "constant"
    else:
        raise ValueError("'trials' must be a variable name or an integer.")

    return Proportion(successes, trials, trials_type)


class Offset:
    def __init__(self, x):
        self.size = None
        if not (is_numeric_dtype(x) or isinstance(x, (int, float))):
            raise ValueError("offset() can only be used with numeric variables.")

        if isinstance(x, pd.Series):
            self.x = x.values
            self.type = "variable"
        elif isinstance(x, (int, float)):
            self.x = x
            self.type = "constant"
        else:
            raise ValueError("'x' must be a variable name or a number.")

    def eval(self):
        if self.type == "variable":
            return self.x.flatten()[:, np.newaxis]
        else:
            return np.ones((self.size, 1)) * self.x

    def set_size(self, size):
        self.size = size


def offset(x):
    return Offset(x)


class BSpline:
    """B-Spline representation

    Generates a B-spline basis for ``x``, allowing non-linear fits. The usual
    usage is something like::

        y ~ 1 + bs(x, 4)

    to fit ``y`` as a smooth function of ``x``, with 4 degrees of freedom
    given to the smooth.

    Parameters
    ----------
    x: 1D array-like
        The data.
    df: The number of degrees of freedom to use for this spline. The return value will have this
        many columns. You must specify at least one of ``df`` and ``knots``.
    knots: 1D array-like or None
        The interior knots to use for the spline. If unspecified, then equally spaced quantiles of
        the input data are used. You must specify at least one of ``df`` and ``knots`
    degree: int
        Degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept: bool
        If ``True``, an intercept is included in the basis. Default is ``False``.
    lower_bound:
        The lower exterior knot location.
    upper_bound:
        The upper exterior knot location.
    """

    def __init__(self):
        self.params_set = False
        self._intercept = None
        self._degree = None
        self._knots = None

    def __call__(
        self, x, df=None, knots=None, degree=3, intercept=False, lower_bound=None, upper_bound=None
    ):
        if not self.params_set:
            self._initialize(x, df, knots, degree, intercept, lower_bound, upper_bound)
        return self.eval(x)

    def _initialize(self, x, df, knots, degree, intercept, lower_bound, upper_bound):

        if not isinstance(degree, int):
            raise ValueError(f"'degree' must be an integer, not {type(degree)}")

        if degree < 0:
            raise ValueError(f"'degree' must be greater than 0, not {degree}")

        if df is None and knots is None:
            raise ValueError("Must specify either 'df' or 'knots'")

        if df and not isinstance(df, int):
            raise ValueError("'df' must be either None or integer")
        # XTODO: Check the type of knots.

        order = degree + 1

        if df is not None:
            n_inner_knots = df - order
            if not intercept:
                n_inner_knots += 1
            if n_inner_knots < 0:
                # We know that n_inner_knots is negative;
                # If df were that much larger, it would have been zero, and things would work.
                raise ValueError(
                    f"df={df} is too small for degree={degree} and intercept={intercept}; "
                    f"it must be >= {df - n_inner_knots}"
                )

            # User specified 'df' AND 'knots'
            if knots is not None:
                if len(knots) != n_inner_knots:
                    raise ValueError(
                        f"df={df} with degree={degree} implies {n_inner_knots} knots; "
                        f"but {len(knots)} were provided"
                    )
            # User specified 'df' but NOT 'knots'
            else:
                knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
                inner_knots = np.percentile(x, 100 * np.asarray(knot_quantiles))

        if knots is not None:
            inner_knots = knots

        if lower_bound is None:
            lower_bound = np.min(x)

        if upper_bound is None:
            upper_bound = np.max(x)

        if lower_bound > upper_bound:
            raise ValueError(f"'lower_bound' > 'upper_bound' ({lower_bound} > {upper_bound})")

        inner_knots = np.asarray(inner_knots)
        if inner_knots.ndim > 1:
            raise ValueError("'knots' must be 1 dimensional")

        if np.any(inner_knots < lower_bound):
            raise ValueError(
                f"Some knot values {inner_knots[inner_knots < lower_bound]} "
                f"fall below lower bound {lower_bound}"
            )

        if np.any(inner_knots > upper_bound):
            raise ValueError(
                f"Some knot values {inner_knots[inner_knots > upper_bound]} "
                f"fall above upper bound {upper_bound}"
            )

        all_knots = np.concatenate(([lower_bound, upper_bound] * order, inner_knots))
        all_knots.sort()

        self._intercept = intercept
        self._degree = degree
        self._knots = all_knots
        self.params_set = True

    def eval(self, x):
        n_bases = len(self._knots) - (self._degree + 1)
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        for i in range(n_bases):
            coefs = np.zeros((n_bases,))
            coefs[i] = 1
            basis[:, i] = splev(x, (self._knots, coefs, self._degree))

        if not self._intercept:
            basis = basis[:, 1:]
        return basis


TRANSFORMS = {
    "I": I,
    "C": C,
    "binary": binary,
    "p": proportion,
    "prop": proportion,
    "proportion": proportion,
    "offset": offset,
}
STATEFUL_TRANSFORMS = {"center": Center, "scale": Scale, "standardize": Scale, "bs": BSpline}
