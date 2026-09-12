# pylint: disable=invalid-name
import inspect

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy.interpolate import CubicSpline as SciPyCubicSpline, splev
from scipy.sparse.linalg import eigsh

from formulae.categorical import CategoricalBox, Sum, Treatment

TRANSFORMS = {}


def is_class_callable(cls):
    members = (member[0] for member in inspect.getmembers(cls))
    return "__call__" in members


# Stateful transformations.
# These transformations have memory about the state of parameters that are
# required to compute the transformation and are obtained as a subproduct of the
# data that is used to compute the transform.
def register_stateful_transform(cls):
    assert isinstance(cls, type), "Can only decorate classes"
    assert is_class_callable(cls), "The class must implement a __call__ method"
    key = cls.__transform_name__ if hasattr(cls, "__transform_name__") else cls.__name__
    cls.__stateful_transform__ = True
    TRANSFORMS[key] = cls
    return cls


@register_stateful_transform
class Center:
    __transform_name__ = "center"

    def __init__(self):
        self.params_set = False
        self.mean = None

    def __call__(self, x):
        if not self.params_set:
            self.mean = np.mean(x)
            self.params_set = True
        return x - self.mean


@register_stateful_transform
class Scale:
    __transform_name__ = "scale"

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
    This is an alias for `{x}`, which does exactly the same, but in a more concise manner.

    Examples
    ----------

    >>> x + I(x**2)
    >>> x + {x**2}
    >>> {(x + y) / z}
    """
    return x


def C(data, contrast=None, levels=None):
    if isinstance(data, CategoricalBox):
        if contrast is None:
            contrast = data.contrast
        if levels is None:
            levels = data.levels
        data = data.data
    return CategoricalBox(data, contrast, levels)


def S(data, omit=None, levels=None):
    """Convert to categorical using Sum encoding

    It is a shorthand for C(x, Sum)
    """
    return CategoricalBox(data, Sum(omit), levels)


def T(data, ref=None, levels=None):
    """Convert to categorical using Treatment encoding

    It is a shorthand for C(x, Treatment)
    """
    return CategoricalBox(data, Treatment(ref), levels)


def binary(x, success=None):
    """Make a variable binary

    Parameters
    ----------
    x : pd.Series
        The object containing the variable to be converted to binary.
    success : str, numeric or None
        The success level. When the variable is equal to this level, the binary variable is 1.
        All the rest are 0. Defaults to `None` which means formulae is going to sort all the
        values in the variable and pick the first one as success.

    Returns
    -------
    x : np.array
        A 0-1 numpy array with shape `(n, 1)` where `n` is the number of observations.
    """
    if success is None:
        categories = sorted(x.unique().tolist())
        success = categories[0]
    booleans = x == success
    if not sum(booleans):
        raise ValueError(f"No value in 'x' is equal to \"{success}\"")
    return np.where(booleans, 1, 0)


class Proportion:
    """Representation of a proportion term.

    Parameters
    ----------
    successes : ndarray
        1D array containing data with `int` dtype.
    trials : ndarray
        1D array containing data with `int` dtype. Its values must be equal or larger than the
        values in `successes`
    trials_type : str
        Indicates whether `trials` is a constant value or not. It can be either `"constant"`
        or `"variable"`.
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
    """Create a term that represents the proportion `successes/trials`.

    This function is actually a wrapper of class `Proportion` that checks its arguments.

    Parameters
    ----------
    successes : pd.Series
        The number of successes for each observation unit.
    trials : pd.Series or int
        The number of trials for each observation unit. If `int`, this function internally
        generates an array of the same length than `successes`.
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
            self.kind = "variable"
        elif isinstance(x, (int, float)):
            self.x = x
            self.kind = "constant"
        else:
            raise ValueError("'x' must be a variable name or a number.")

    def eval(self):
        if self.kind == "variable":
            return self.x.flatten()
        else:
            return np.ones((self.size, 1)) * self.x

    def set_size(self, size):
        self.size = size


def offset(x):
    return Offset(x)


@register_stateful_transform
class BSpline:
    """B-Spline representation

    Generates a B-spline basis for `x`, allowing non-linear fits. The usual
    usage is something like::

        y ~ 1 + bs(x, 4)

    to fit `y` as a smooth function of `x`, with 4 degrees of freedom
    given to the smooth.

    Parameters
    ----------
    x : 1D array-like
        The data.
    df : The number of degrees of freedom to use for this spline. The return value will have this
        many columns. You must specify at least one of `df` and `knots`.
    knots : 1D array-like or None
        The interior knots to use for the spline. If unspecified, then equally spaced quantiles of
        the input data are used. You must specify at least one of `df` and `knots`
    degree : int
        Degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool
        If `True`, an intercept is included in the basis. Default is `False`.
    lower_bound :
        The lower exterior knot location.
    upper_bound :
        The upper exterior knot location.
    """

    __transform_name__ = "bs"

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

        # NOTE: We need to clean the logic that creates 'inner_knots'.
        inner_knots = np.asarray(inner_knots)  # pylint: disable=used-before-assignment
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


class _CubicRegressionSpline:
    """Shared implementation for natural and cyclic cubic regression splines."""

    def __init__(self):
        self.params_set = False
        self._lower_bound = None
        self._upper_bound = None
        self._knots = None
        self._center = None
        self._centering_matrix = None
        self._spline = None

    @staticmethod
    def _get_knots(
        df,
        knots,
        lower_bound,
        upper_bound,
        data,
        df_offset,
        minimum_inner_knots,
    ):
        if df is not None and not isinstance(df, int):
            raise ValueError("'df' must be either None or an integer")

        minimum_df = df_offset + minimum_inner_knots
        if df is not None and df < minimum_df:
            raise ValueError(f"'df' must be greater than or equal to {minimum_df}")

        if knots is not None:
            inner_knots = np.asarray(knots, dtype=float)
            if inner_knots.ndim != 1:
                raise ValueError("'knots' must be one dimensional")

            if not np.all(np.isfinite(inner_knots)):
                raise ValueError("'knots' must contain only finite values")

            if np.unique(inner_knots).size != inner_knots.size:
                raise ValueError("'knots' must not contain repeated values")

            if np.any(inner_knots <= lower_bound) or np.any(inner_knots >= upper_bound):
                raise ValueError("All knots must fall strictly between the boundaries")

            if inner_knots.size < minimum_inner_knots:
                raise ValueError(
                    f"This cubic spline requires at least {minimum_inner_knots} interior knot(s)"
                )

            inferred_df = inner_knots.size + df_offset
            if df is not None and df != inferred_df:
                raise ValueError(
                    f"df={df} implies {df - df_offset} knots, but {inner_knots.size} were provided"
                )
            inner_knots = np.sort(inner_knots)
        else:
            n_inner_knots = df - df_offset
            knot_quantiles = np.linspace(0, 1, n_inner_knots + 2)[1:-1]
            knot_data = np.unique(data)
            inner_knots = np.quantile(knot_data, knot_quantiles)

        return inner_knots

    def _set_state(self, spline, all_knots, raw_basis, center):
        centering_matrix = None

        if center:
            column_means = raw_basis.mean(axis=0)
            # Keep only directions orthogonal to the column means. This makes the fitted smooth
            # average to zero on the training data and removes the constant direction, which is
            # represented by the model intercept.
            unit_means = column_means / np.linalg.norm(column_means)
            reflector = unit_means.copy()
            reflector[0] += 1 if unit_means[0] >= 0 else -1
            reflector /= np.linalg.norm(reflector)
            householder = np.eye(column_means.size) - 2 * np.outer(reflector, reflector)
            centering_matrix = householder[:, 1:]

        self._lower_bound = all_knots[0]
        self._upper_bound = all_knots[-1]
        self._knots = all_knots
        self._center = center
        self._centering_matrix = centering_matrix
        self._spline = spline
        self.params_set = True

    @property
    def bounds(self):
        """Remembered lower and upper boundary knots."""
        return self._lower_bound, self._upper_bound


@register_stateful_transform
class CyclicCubicSpline(_CubicRegressionSpline):
    """Cyclic cubic regression spline representation.

    Generates a cubic spline basis for periodic covariates. The values and first two derivatives of
    every basis function agree at the two boundaries, so any linear combination of the columns
    joins smoothly across the boundary.

    Parameters
    ----------
    x : 1D array-like
        The data.
    period : int or float
        Length of the cycle. It must be finite and greater than zero.
    df : int or None
        Number of columns in the returned basis, after applying the centering constraint. Defaults
        to 10. This differs from the `k` argument in mgcv, which counts dimensions before all
        constraints are absorbed. If `knots` is supplied without `df`, the number of columns is
        inferred from the knots.
    knots : 1D array-like or None
        Interior knots. They must be finite, distinct, and strictly between the boundaries.
        If omitted, knots are placed at equally spaced quantiles of the unique observed phases.
    lower_bound : float or None
        Start of the cycle. Defaults to the minimum observed value. The end of the cycle is computed
        as `lower_bound + period`.
    center : bool
        If `True` (the default), center the spline basis using the original data. This avoids
        confounding the smooth with a model intercept.

    Notes
    -----
    Values outside the boundaries, including new data, are wrapped into the original interval.
    With centering enabled, `df=d` corresponds to `k=d+2` in a centered mgcv cyclic smooth: mgcv
    also counts the periodic endpoint identification before absorbing the centering constraint.
    Use the default centered basis in a model with an intercept. For a model without an intercept,
    use `center=False` if the spline must also represent the overall constant.
    """

    __transform_name__ = "cc"

    def __init__(self):
        super().__init__()
        self._period = None

    def __call__(
        self,
        x,
        period,
        df=None,
        knots=None,
        lower_bound=None,
        center=True,
    ):
        if not self.params_set:
            self._initialize(x, period, df, knots, lower_bound, center)
        return self.eval(x)

    def _initialize(self, x, period, df, knots, lower_bound, center):
        x = np.asarray(x)

        if not isinstance(period, (int, float)):
            raise ValueError("'period' must be a number")

        if period <= 0 or not np.isfinite(period):
            raise ValueError("'period' must be finite and greater than zero")

        if lower_bound is not None and not isinstance(lower_bound, (int, float)):
            raise ValueError("'lower_bound' must be a finite number")

        period = float(period)
        center = bool(center)
        lower_bound = float(np.min(x) if lower_bound is None else lower_bound)
        upper_bound = lower_bound + period

        # Keep `cc(x, period=...)` convenient while allowing knots to determine the basis
        # dimension when `df` is explicitly or implicitly omitted.
        if df is None and knots is None:
            df = 10

        wrapped_x = lower_bound + np.mod(x - lower_bound, period)
        df_offset = 0 if center else 1
        inner_knots = self._get_knots(
            df,
            knots,
            lower_bound,
            upper_bound,
            wrapped_x,
            df_offset,
            minimum_inner_knots=1,
        )
        all_knots = np.concatenate(([lower_bound], inner_knots, [upper_bound]))
        n_free = all_knots.size - 1

        # The endpoint value is not independent for a periodic spline:
        # the final row repeats the first row of the identity matrix.
        values = np.vstack((np.eye(n_free), np.eye(n_free)[0]))
        spline = SciPyCubicSpline(
            all_knots, values, axis=0, bc_type="periodic", extrapolate="periodic"
        )

        raw_basis = spline(wrapped_x)
        self._period = period
        self._set_state(spline, all_knots, raw_basis, center)

    def eval(self, x):
        x = np.asarray(x)
        wrapped_x = self._lower_bound + np.mod(x - self._lower_bound, self._period)
        basis = self._spline(wrapped_x)

        if self._center:
            basis = basis @ self._centering_matrix

        return basis

    @property
    def period(self):
        """Length of the remembered cycle."""
        return self._period


@register_stateful_transform
class NaturalCubicSpline(_CubicRegressionSpline):
    """Natural cubic regression spline representation.

    Generates a cubic spline basis satisfying `f''(a) = f''(b) = 0`.
    Values outside the boundary knots are evaluated using exact linear continuation,
    rather than cubic polynomial extrapolation.

    Parameters
    ----------
    x : 1D array-like
        The data.
    df : int or None
        Number of columns in the returned basis, after applying the centering constraint. Defaults
        to 10. This differs from the `k` argument in mgcv, which counts dimensions before the
        centering constraint is absorbed. If `knots` is supplied without `df`, the number of
        columns is inferred from the knots.
    knots : 1D array-like or None
        Interior knots. If omitted, knots are placed at equally spaced quantiles of the unique
        observed values.
    lower_bound : float or None
        Lower boundary knot. Defaults to the minimum observed value.
    upper_bound : float or None
        Upper boundary knot. Defaults to the maximum observed value.
    center : bool
        If `True` (the default), center the spline basis using the original data.

    Notes
    -----
    With centering enabled, `df=d` corresponds to `k=d+1` in a centered mgcv natural cubic smooth.
    Use the default centered basis in a model with an intercept. For a model without an intercept,
    use `center=False` if the spline must also represent the overall constant.
    """

    __transform_name__ = "cr"

    def __call__(
        self,
        x,
        df=None,
        knots=None,
        lower_bound=None,
        upper_bound=None,
        center=True,
    ):
        if not self.params_set:
            self._initialize(x, df, knots, lower_bound, upper_bound, center)
        return self.eval(x)

    def _initialize(self, x, df, knots, lower_bound, upper_bound, center):
        for name, value in (("lower_bound", lower_bound), ("upper_bound", upper_bound)):
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"'{name}' must be a number")

        if df is None and knots is None:
            df = 10

        x = np.asarray(x)
        center = bool(center)
        lower_bound = float(np.min(x) if lower_bound is None else lower_bound)
        upper_bound = float(np.max(x) if upper_bound is None else upper_bound)

        if lower_bound >= upper_bound:
            raise ValueError("'lower_bound' must be less than 'upper_bound'")

        df_offset = 1 if center else 2
        inner_knots = self._get_knots(
            df,
            knots,
            lower_bound,
            upper_bound,
            x,
            df_offset,
            minimum_inner_knots=0,
        )
        all_knots = np.concatenate(([lower_bound], inner_knots, [upper_bound]))
        values = np.eye(all_knots.size)
        spline = SciPyCubicSpline(all_knots, values, axis=0, bc_type="natural", extrapolate=False)

        raw_basis = self._eval_linear_tails(x, spline, lower_bound, upper_bound)
        self._set_state(spline, all_knots, raw_basis, center)

    @staticmethod
    def _eval_linear_tails(x, spline, lower_bound, upper_bound):
        """Evaluate inside the knot range and continue with tangent lines outside."""
        basis = spline(np.clip(x, lower_bound, upper_bound))
        below = x < lower_bound
        above = x > upper_bound

        if np.any(below):
            basis[below] = spline(lower_bound) + np.multiply.outer(
                x[below] - lower_bound, spline(lower_bound, nu=1)
            )

        if np.any(above):
            basis[above] = spline(upper_bound) + np.multiply.outer(
                x[above] - upper_bound, spline(upper_bound, nu=1)
            )

        return basis

    def eval(self, x):
        x = np.asarray(x)
        basis = self._eval_linear_tails(x, self._spline, self._lower_bound, self._upper_bound)

        if self._center:
            basis = basis @ self._centering_matrix

        return basis


@register_stateful_transform
class ThinPlateRegressionSpline:
    """Low-rank univariate thin-plate regression spline.

    This implements the rank-reduced thin-plate construction of Wood (2003).
    For a univariate, second-order smooth the radial kernel is `abs(x - x_i) ** 3 / 12` and
    the penalty null space is spanned by a constant and a linear function.

    Parameters
    ----------
    x : 1D array-like
        The data.
    df : int
        Number of columns in the returned basis, after centering. Defaults to 10. At least two
        columns are required: one unpenalized linear column and one penalized column.
    center : bool
        If `True` (the default), impose a sum-to-zero constraint over the training data and omit
        the constant null-space column. Use this form in a model containing an intercept. With
        `center=False`, the constant and linear null-space columns are both retained and `df`
        must be at least three.
    max_knots : int
        Maximum number of unique data locations used in the eigendecomposition. If there are more
        unique locations, a deterministic subsample is used. These internal locations are not
        user-selected regression-spline knots. Defaults to 2000, as in mgcv.
    seed : int
        Seed used only when subsampling more than `max_knots` unique locations. Defaults to 1.

    Notes
    -----
    The covariate is shifted and scaled before constructing the kernel. This improves numerical
    conditioning and makes the whitened smoothing prior invariant to the units used for `x`.
    Columns are returned with the penalty null space first. The :attr:`penalty` is expressed in
    the returned coordinates and, by construction, is diagonal: the null-space columns have zero
    penalty and the remaining columns have unit penalty. Consequently a smoothing prior can give
    the penalized coefficients a shared scale while assigning the null-space coefficients their
    own weakly informative prior. `formulae` constructs design matrices and does not itself
    create coefficient priors; downstream modelling packages can obtain this information from
    the remembered stateful transform.

    New data are evaluated against the locations, eigenspace, constraints, scaling, and centering
    values learned from the training data. No extrapolation rule or new knots are selected.
    """

    __transform_name__ = "tp"
    _NULL_SPACE_DIMENSION = 2

    def __init__(self):
        self.params_set = False
        self._center = None
        self._df = None
        self._shift = None
        self._scale = None
        self._sites = None
        self._radial_map = None
        self._column_means = None
        self._penalty = None
        self._null_space_dimension = None

    def __call__(self, x, df=10, center=True, max_knots=2000, seed=1):
        if not self.params_set:
            self._initialize(x, df, center, max_knots, seed)
        return self.eval(x)

    @staticmethod
    def _validate_x(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("'x' must be one dimensional")
        if x.size == 0:
            raise ValueError("'x' must contain at least one value")
        if not np.all(np.isfinite(x)):
            raise ValueError("'x' must contain only finite values")
        return x

    @staticmethod
    def _kernel(x, sites):
        return np.abs(np.subtract.outer(x, sites)) ** 3 / 12

    @staticmethod
    def _truncated_eigendecomposition(matrix, rank):
        """Return eigenpairs ordered by decreasing eigenvalue magnitude."""
        size = matrix.shape[0]
        if rank >= size // 2 or size <= 200:
            values, vectors = np.linalg.eigh(matrix)
        else:
            # A fixed initial vector makes ARPACK results reproducible across calls.
            values, vectors = eigsh(matrix, k=rank, which="LM", v0=np.ones(size))

        order = np.argsort(np.abs(values))[::-1][:rank]
        return values[order], vectors[:, order]

    def _initialize(self, x, df, center, max_knots, seed):
        x = self._validate_x(x)

        if not isinstance(df, int) or isinstance(df, bool):
            raise ValueError("'df' must be an integer")
        if not isinstance(max_knots, int) or isinstance(max_knots, bool):
            raise ValueError("'max_knots' must be an integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("'seed' must be an integer")
        if max_knots < self._NULL_SPACE_DIMENSION + 1:
            raise ValueError("'max_knots' must be greater than the penalty null-space dimension")

        center = bool(center)
        minimum_df = 2 if center else 3
        if df < minimum_df:
            raise ValueError(f"'df' must be greater than or equal to {minimum_df}")

        # Centering absorbs the constant constraint, so the pre-constraint TPRS rank is one larger
        # than the number of returned columns.
        full_rank = df + int(center)
        shift = np.mean(x)
        scale = np.sqrt(np.mean((x - shift) ** 2))
        if scale == 0:
            raise ValueError("'x' must contain at least two distinct values")
        standardized_x = (x - shift) / scale
        sites = np.unique(standardized_x)

        if sites.size > max_knots:
            # RandomState, rather than default_rng, preserves support for NumPy 1.16.
            rng = np.random.RandomState(seed)  # pylint: disable=no-member
            sites = np.sort(rng.choice(sites, size=max_knots, replace=False))

        if sites.size < full_rank:
            raise ValueError(
                f"A thin-plate basis with df={df} requires at least {full_rank} unique "
                f"values, but only {sites.size} were found"
            )

        kernel = self._kernel(sites, sites)
        eigenvalues, eigenvectors = self._truncated_eigendecomposition(kernel, full_rank)

        polynomial = np.column_stack((np.ones(sites.size), sites))
        constraints = eigenvectors.T @ polynomial
        constraint_q, _ = np.linalg.qr(constraints, mode="complete")
        constraint_basis = constraint_q[:, self._NULL_SPACE_DIMENSION :]

        # In the paper's coordinates the radial design is E U Z and its penalty is Z' D Z.
        # Whiten that positive-definite penalty so a spherical coefficient prior controls the
        # wiggliness directly.
        raw_radial_map = eigenvectors @ constraint_basis
        raw_penalty = constraint_basis.T @ (eigenvalues[:, None] * constraint_basis)
        raw_penalty = (raw_penalty + raw_penalty.T) / 2
        penalty_values, penalty_vectors = np.linalg.eigh(raw_penalty)
        tolerance = np.spacing(1.0) * max(raw_penalty.shape) * np.max(np.abs(penalty_values))
        if np.any(penalty_values <= tolerance):
            raise ValueError(
                "Could not construct a positive-definite thin-plate penalty; "
                "try a smaller value of 'df'"
            )
        whitening = penalty_vectors / np.sqrt(penalty_values)
        radial_map = raw_radial_map @ whitening

        radial_basis = self._kernel(standardized_x, sites) @ radial_map
        if center:
            raw_basis = np.column_stack((standardized_x, radial_basis))
            column_means = raw_basis.mean(axis=0)
            null_space_dimension = 1
        else:
            raw_basis = np.column_stack((np.ones(x.size), standardized_x, radial_basis))
            column_means = np.zeros(raw_basis.shape[1])
            null_space_dimension = 2

        penalty = np.diag(
            np.concatenate(
                (np.zeros(null_space_dimension), np.ones(raw_basis.shape[1] - null_space_dimension))
            )
        )

        self._center = center
        self._df = df
        self._shift = shift
        self._scale = scale
        self._sites = sites
        self._radial_map = radial_map
        self._column_means = column_means
        self._penalty = penalty
        self._null_space_dimension = null_space_dimension
        self.params_set = True

    def eval(self, x):
        x = self._validate_x(x)
        standardized_x = (x - self._shift) / self._scale
        radial_basis = self._kernel(standardized_x, self._sites) @ self._radial_map

        if self._center:
            basis = np.column_stack((standardized_x, radial_basis))
        else:
            basis = np.column_stack((np.ones(x.size), standardized_x, radial_basis))

        return basis - self._column_means

    @property
    def penalty(self):
        """Quadratic wiggliness penalty in the returned basis coordinates."""
        return self._penalty.copy()

    @property
    def null_space_dimension(self):
        """Number of leading, unpenalized columns in the returned basis."""
        return self._null_space_dimension

    @property
    def rank(self):
        """Rank of the wiggliness penalty."""
        return self._df - self._null_space_dimension

    @property
    def sites(self):
        """Standardized unique data locations used to construct the low-rank eigenspace."""
        return self._sites.copy()


@register_stateful_transform
class Polynomial:
    """Polynomial transformation

    The computation of this transformation is borrowed from the implementation in the
    Formulaic library written by Matthew Wardrop.

    The original implementation and more documentation can be found here:
    https://github.com/matthewwardrop/formulaic/blob/main/formulaic/transforms/poly.py

    Parameters
    ----------
    x : 1d array-like
        The data.
    degree : int
        The degree of the polynomial terms to compute. If degree is k, with k > 1, this
        transformation computes the polynomials x^1, x^2, ...x^k.
    raw : bool
        Whether to use raw polynomials or orthonormal ones. Defaults to False.
    """

    __transform_name__ = "poly"

    def __init__(self):
        self.params_set = False
        self.degree = 1
        self.raw = False
        self.alpha = {}
        self.norms2 = {}

    def __call__(self, x, degree=1, raw=False):
        if not self.params_set:
            self.degree = degree
            self.raw = raw
        return self.eval(x)

    def eval(self, x):
        if self.raw:
            return np.column_stack([np.power(x, k) for k in range(1, self.degree + 1)])

        def get_alpha(k):
            if k not in self.alpha:
                self.alpha[k] = np.sum(x * P[:, k] ** 2) / np.sum(P[:, k] ** 2)
            return self.alpha[k]

        def get_norm(k):
            if k not in self.norms2:
                self.norms2[k] = np.sum(P[:, k] ** 2)
            return self.norms2[k]

        def get_beta(k):
            return get_norm(k) / get_norm(k - 1)

        P = np.empty((x.shape[0], self.degree + 1))
        P[:, 0] = 1

        for i in range(1, self.degree + 1):
            P[:, i] = (x - get_alpha(i - 1)) * P[:, i - 1]
            if i >= 2:
                P[:, i] -= get_beta(i - 1) * P[:, i - 2]

        P /= np.array([np.sqrt(get_norm(k)) for k in range(0, self.degree + 1)])
        return P[:, 1:]


TRANSFORMS.update(
    {
        "B": binary,
        "binary": binary,
        "C": C,
        "I": I,
        "offset": offset,
        "p": proportion,
        "prop": proportion,
        "proportion": proportion,
        "S": S,
        "standardize": Scale,
        "T": T,
    }
)
