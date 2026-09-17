# pylint: disable=invalid-name,too-many-lines
import inspect
import warnings

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from scipy.interpolate import CubicSpline as SciPyCubicSpline, splev
from scipy.sparse.linalg import eigsh

from formulae.categorical import CategoricalBox, Sum, Treatment
from formulae.utils import get_centering_matrix

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

    Generates a B-spline basis for non-linear fits. For example, `y ~ 1 + bs(x, df=4)`
    gives four centered columns for the smooth, alongside the model intercept.

    Parameters
    ----------
    x : 1D array-like
        The data.
    df : int or None
        Number of columns in the returned basis, after applying the centering constraint.
        You must specify at least one of `df` and `knots`.
    knots : 1D array-like or None
        The interior knots to use for the spline. If unspecified, then equally spaced quantiles of
        the input data are used. You must specify at least one of `df` and `knots`
    degree : int
        Degree of the piecewise polynomial. Default is 3 for cubic splines.
    intercept : bool or None
        Deprecated alias for `center=not intercept`. Defaults to `None`. Passing a boolean emits
        a warning and overrides `center`. In particular, `intercept=False` now centers the full
        basis instead of dropping its first column.
    lower_bound :
        The lower exterior knot location.
    upper_bound :
        The upper exterior knot location.
    center : bool
        If `True` (the default), impose a sum-to-zero constraint over the training data. The
        constraint is absorbed into the basis and reused for new data. Use `center=False` to
        retain the full basis, including the ability to represent a constant.
    """

    __transform_name__ = "bs"

    def __init__(self):
        self.params_set = False
        self._center = None
        self._centering_matrix = None
        self._degree = None
        self._knots = None

    def __call__(
        self,
        x,
        df=None,
        knots=None,
        degree=3,
        intercept=None,
        lower_bound=None,
        upper_bound=None,
        center=True,
    ):
        if intercept is not None:
            center = not intercept
            warnings.warn(
                "'intercept' is deprecated; use 'center' instead. "
                f"Using center={center}. Centering replaces dropping the first basis column "
                "when intercept=False.",
                FutureWarning,
                stacklevel=2,
            )

        if not self.params_set:
            self._initialize(x, df, knots, degree, center, lower_bound, upper_bound)
        return self.eval(x)

    def _initialize(self, x, df, knots, degree, center, lower_bound, upper_bound):

        if not isinstance(degree, int):
            raise ValueError(f"'degree' must be an integer, not {type(degree)}")

        if degree < 0:
            raise ValueError(f"'degree' must be greater than 0, not {degree}")

        if df is None and knots is None:
            raise ValueError("Must specify either 'df' or 'knots'")

        if df and not isinstance(df, int):
            raise ValueError("'df' must be either None or integer")

        order = degree + 1
        center = bool(center)

        if df is not None:
            n_inner_knots = df - order
            if center:
                n_inner_knots += 1
            if n_inner_knots < 0:
                raise ValueError(
                    f"df={df} is too small for degree={degree} and center={center}; "
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

        self._center = center
        self._degree = degree
        self._knots = all_knots
        if center:
            self._centering_matrix = get_centering_matrix(self._eval_basis(x))
        self.params_set = True

    def _eval_basis(self, x):
        x = np.asarray(x)
        n_bases = len(self._knots) - (self._degree + 1)
        basis = np.empty((x.shape[0], n_bases), dtype=float)
        for i in range(n_bases):
            coefs = np.zeros((n_bases,))
            coefs[i] = 1
            basis[:, i] = splev(x, (self._knots, coefs, self._degree))

        return basis

    def eval(self, x):
        basis = self._eval_basis(x)
        if self._center:
            basis = basis @ self._centering_matrix
        return basis


class _RandomEffectsSplineMixin:
    """Stateful conversion of a single-penalty spline to random effects."""

    def to_random(self, basis=None):
        """Return the fixed/null-space columns followed by whitened random-effect columns.

        Parameters
        ----------
        basis : 2D array-like or None
            A basis returned by this fitted transform (for example `spline.eval(x_new)`).
            Omit to convert the original training basis.

        Returns
        -------
        ndarray
            A new matrix with the same shape as the input basis. Its first
            `null_space_dimension` columns are unpenalized by the smoothing prior.
            For CR and TP these are a column of ones followed by the standardized linear
            covariate when `center=False`, or just the standardized linear covariate
            when centered. Standardization uses the training mean and population standard
            deviation. CC has only a constant when uncentered and no null columns when centered.

        Notes
        -----
        The transformed penalty is `diag(0, I)`: curved coefficients can share independent
        `Normal(0, tau)` priors, while null-space coefficients require separate priors.
        This reparameterizes the same curvature penalty without changing its value.
        This is analogous to `mgcv::smooth2random(type=2)`, but returns one matrix with null
        columns first, rather than separate fixed and random matrices.
        Unlike mgcv's default `smoothCon`, penalties are not rescaled for numerical convenience.

        Examples
        --------
        >>> spline = NaturalCubicSpline()
        >>> B = spline(np.linspace(0, 1, 20), df=5)
        >>> Z = spline.to_random()
        >>> Z_new = spline.to_random(spline.eval([0.2, 1.2]))
        """
        if not self.params_set:
            raise ValueError("Fit the spline before calling 'to_random'")

        state = self._random_state
        basis = state["basis"] if basis is None else np.asarray(basis, dtype=float)

        if state["transform"] is None:
            values, vectors = np.linalg.eigh(self.penalty_matrix)
            m = self.null_space_dimension
            positive = values[m:]
            tolerance = np.spacing(1.0) * len(values) * np.max(np.abs(values))

            if np.any(positive <= tolerance):
                raise ValueError("The spline penalty is numerically singular. Try a smaller 'df'")

            curved = vectors[:, m:] / np.sqrt(positive)
            state["transform"] = np.column_stack((state["null"], curved))

        return basis @ state["transform"]

    def _set_random_state(self, basis, null_coefficients):
        self._random_state = {  # pylint: disable=attribute-defined-outside-init
            "basis": basis.copy(),
            "null": null_coefficients,
            "transform": None,
        }


class _CubicRegressionSpline(_RandomEffectsSplineMixin):
    """Shared implementation for natural and cyclic cubic regression splines."""

    _RAW_NULL_SPACE_DIMENSION = 2

    def __init__(self):
        self.params_set = False
        self._lower_bound = None
        self._upper_bound = None
        self._knots = None
        self._center = None
        self._centering_matrix = None
        self._spline = None
        self._penalty_matrix = None
        self._random_state = None

    @staticmethod
    def _get_knots(
        df,
        knots,
        lower_bound,
        upper_bound,
        data,
        df_offset,
        minimum_inner_knots,
        cyclic,
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
            knot_data = np.unique(data[(data >= lower_bound) & (data <= upper_bound)])

            if n_inner_knots and knot_data.size < n_inner_knots + (1 if cyclic else 2):
                raise ValueError("Not enough unique values within the boundaries for this 'df'")

            inner_knots = np.quantile(knot_data, knot_quantiles) if n_inner_knots else np.array([])

        required = inner_knots.size + (1 if cyclic else 2)
        if np.unique(data).size < required:
            raise ValueError(f"This spline requires at least {required} unique values")

        return inner_knots

    def _set_state(self, spline, all_knots, raw_basis, center):
        centering_matrix = None

        if np.linalg.matrix_rank(raw_basis) < raw_basis.shape[1]:
            raise ValueError("The spline basis is rank deficient on the training data")

        # Second derivatives are linear on each interval.
        # Two-point Gauss quadrature therefore integrates their pairwise products exactly.
        widths = np.diff(all_knots)
        midpoints = (all_knots[:-1] + all_knots[1:]) / 2
        points = midpoints[:, None] + widths[:, None] * np.array([-1, 1]) / (2 * np.sqrt(3))
        derivatives = spline(points.ravel(), nu=2)

        if center:
            # Absorb the training sum-to-zero constraint, leaving the constant to the intercept.
            centering_matrix = get_centering_matrix(raw_basis)
            derivatives = derivatives @ centering_matrix

        penalty = derivatives.T @ (np.repeat(widths / 2, 2)[:, None] * derivatives)

        # Cardinal coefficients for explicit constant/linear directions.
        # Centering projects the mean-zero linear direction into the returned coordinates.
        null = np.ones((raw_basis.shape[1], 1))
        if self._RAW_NULL_SPACE_DIMENSION == 2:
            # Natural cardinal splines reproduce the covariate, including linear tails.
            x = raw_basis @ all_knots
            linear = (all_knots - x.mean()) / x.std()
            null = np.column_stack((null, linear))
        basis = raw_basis
        if center:
            basis = raw_basis @ centering_matrix
            null = centering_matrix.T @ null[:, 1:]
        self._set_random_state(basis, null)

        self._lower_bound = all_knots[0]
        self._upper_bound = all_knots[-1]
        self._knots = all_knots
        self._center = center
        self._centering_matrix = centering_matrix
        self._spline = spline
        self._penalty_matrix = (penalty + penalty.T) / 2
        self.params_set = True

    @property
    def bounds(self):
        """Remembered lower and upper boundary knots."""
        return self._lower_bound, self._upper_bound

    @property
    def penalty_matrix(self):
        """Integrated squared-curvature penalty in the returned basis coordinates.

        For coefficients `beta`, `beta.T @ S @ beta` equals the integral of `f''(x)**2` over the
        boundary interval, in the original units of `x`.
        For cyclic splines this is the integral over one period.
        Natural spline tails are linear and contribute zero.
        """
        return self._penalty_matrix.copy()

    @property
    def null_space_dimension(self):
        """Dimension of the curvature penalty null space.

        This counts leading columns only in the result of `to_random`.
        """
        return self._RAW_NULL_SPACE_DIMENSION - int(self._center)


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
    `penalty_matrix` supplies the integrated squared second-derivative penalty over one
    period. Its null space is constant when uncentered and empty when centered.
    """

    __transform_name__ = "cc"
    _RAW_NULL_SPACE_DIMENSION = 1

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
            cyclic=True,
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
        observed values within the boundary interval. There must be enough distinct values
        in that interval to place the requested knots.
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
    `penalty_matrix` supplies the integrated squared second-derivative penalty.
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
            cyclic=False,
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
class ThinPlateRegressionSpline(_RandomEffectsSplineMixin):
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
        columns are required: one null-space direction and one penalized direction.
    center : bool
        If `True` (the default), impose a sum-to-zero constraint over the training data and omit
        the constant direction. Use this form in a model containing an intercept. With
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
    Before absorbing the centering constraint, radial columns have unit root mean square over
    training data and are followed by constant and standardized linear columns.
    The full sum-to-zero constraint is absorbed when centered. Eigenvector signs and rotations can
    differ from mgcv; the function space and quadratic penalty are equivalent when the same
    locations are used.

    The covariate is standardized internally for numerical conditioning. `penalty_matrix`
    accounts for this and measures integrated squared curvature in the original units of x,
    without mgcv's optional penalty rescaling. Use `to_random()` explicitly to separate the
    null space and whiten the penalty. Its first `null_space_dimension` columns are then the
    unpenalized directions, with independent, equally penalized curved directions following.

    New data are evaluated against the locations, eigenspace, constraints, scaling, and centering
    values learned from the training data.
    """

    __transform_name__ = "tp"
    _NULL_SPACE_DIMENSION = 2

    def __init__(self):
        self.params_set = False
        self._center = None
        self._shift = None
        self._scale = None
        self._sites = None
        self._radial_map = None
        self._centering_matrix = None
        self._null_space_dimension = None
        self._penalty_matrix = None
        self._random_state = None

    def __call__(self, x, df=10, center=True, max_knots=2000, seed=1):
        if not self.params_set:
            self._initialize(x, df, center, max_knots, seed)
        return self.eval(x)

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
        x = np.asarray(x, dtype=float)

        if not isinstance(df, int):
            raise ValueError("'df' must be an integer")

        if not isinstance(max_knots, int):
            raise ValueError("'max_knots' must be an integer")

        if not isinstance(seed, int):
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

        # Retain the regression coordinates; whitening is an explicit to_random operation.
        raw_radial_map = eigenvectors @ constraint_basis
        raw_penalty = constraint_basis.T @ (eigenvalues[:, None] * constraint_basis)
        raw_penalty = (raw_penalty + raw_penalty.T) / 2
        penalty_values = np.linalg.eigvalsh(raw_penalty)
        tolerance = np.spacing(1.0) * max(raw_penalty.shape) * np.max(np.abs(penalty_values))

        if np.any(penalty_values <= tolerance):
            raise ValueError(
                "Could not construct a positive-definite thin-plate penalty; "
                "try a smaller value of 'df'"
            )
        radial_basis = self._kernel(standardized_x, sites) @ raw_radial_map
        rms = np.sqrt(np.mean(radial_basis**2, axis=0))
        radial_map = raw_radial_map / rms
        raw_basis = np.column_stack((radial_basis / rms, np.ones(x.size), standardized_x))
        penalty = np.zeros((full_rank, full_rank))
        penalty[:-2, :-2] = raw_penalty / np.outer(rms, rms) / scale**3
        null = np.eye(full_rank)[:, -2:]
        centering_matrix = None

        if center:
            centering_matrix = get_centering_matrix(raw_basis)
            raw_basis = raw_basis @ centering_matrix
            penalty = centering_matrix.T @ penalty @ centering_matrix
            null = centering_matrix.T @ null[:, 1:]

        self._center = center
        self._shift = shift
        self._scale = scale
        self._sites = sites
        self._radial_map = radial_map
        self._centering_matrix = centering_matrix
        self._null_space_dimension = 2 - int(center)
        self._penalty_matrix = (penalty + penalty.T) / 2
        self._set_random_state(raw_basis, null)
        self.params_set = True

    def eval(self, x):
        x = np.asarray(x, dtype=float)
        standardized_x = (x - self._shift) / self._scale
        radial_basis = self._kernel(standardized_x, self._sites) @ self._radial_map

        basis = np.column_stack((radial_basis, np.ones(x.size), standardized_x))
        if self._center:
            basis = basis @ self._centering_matrix
        return basis

    @property
    def penalty_matrix(self):
        """Integrated squared-curvature penalty in the default basis, in original x units."""
        return self._penalty_matrix.copy()

    @property
    def null_space_dimension(self):
        """Number of null-space directions, placed first only by `to_random`.

        This is 1 (linear) with centering or 2 (constant, linear) without centering.
        The remaining `to_random` columns have identity curvature penalty.
        """
        return self._null_space_dimension

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
