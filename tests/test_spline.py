import pytest
import re

import numpy as np
import pandas as pd

from formulae.matrices import design_matrices
from formulae.transforms import BSpline


@pytest.fixture
def data():
    rng = np.random.default_rng(1234)
    size = 21
    data = pd.DataFrame(
        {"seq": np.linspace(0, 1, 21), "x1": rng.uniform(size=size), "x2": rng.uniform(size=size)}
    )
    return data


@pytest.fixture
def sequence():
    return np.linspace(0, 1, 21)


def test_basic(sequence):
    matrix = BSpline()(sequence, df=4, center=False)
    # A cubic B-spline without interior knots is the Bernstein polynomial basis.
    expected = np.column_stack(
        (
            (1 - sequence) ** 3,
            3 * sequence * (1 - sequence) ** 2,
            3 * sequence**2 * (1 - sequence),
            sequence**3,
        )
    )
    assert np.allclose(matrix, expected)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("specify_df", [False, True])
def test_bs_centering(degree, specify_df):
    x = np.linspace(0, 1, 41) ** 2
    knots = [0.2, 0.65]
    df = degree + 2
    kwargs = {"df": df} if specify_df else {}
    centered = BSpline()(x, knots=knots, degree=degree, **kwargs)
    full = BSpline()(x, knots=knots, degree=degree, center=False)

    assert centered.shape == (x.size, df)
    assert np.allclose(centered.mean(axis=0), 0)
    assert np.linalg.matrix_rank(centered) == df
    # Adding the intercept must recover exactly the original spline space.
    model = np.column_stack((np.ones(x.size), centered))
    assert np.allclose(model @ np.linalg.pinv(model), full @ np.linalg.pinv(full))


def test_degree(sequence):
    matrix = BSpline()(sequence, df=1, degree=1)
    # The orthonormal centering transformation preserves the scale of the full basis.
    expected = np.sqrt(2) * (sequence - sequence.mean())
    assert np.allclose(matrix[:, 0], expected)


def test_basic_through_design_matrices(data):
    dm = design_matrices("bs(seq, 3) - 1", data)
    matrix = dm.common.design_matrix
    assert np.allclose(matrix, BSpline()(data.seq, df=3))
    assert np.allclose(matrix.mean(axis=0), 0)
    labels = dm.common.terms["bs(seq, 3)"].labels
    assert labels == ["bs(seq, 3)[0]", "bs(seq, 3)[1]", "bs(seq, 3)[2]"]


def test_bs_new_data_preserves_training_constraint(data):
    dm = design_matrices("bs(seq, 1, degree=1)", data)
    new_data = pd.DataFrame({"seq": [-0.5, 0.0, 0.25, 1.5]})
    predicted = dm.common.evaluate_new_data(new_data).design_matrix
    expected = np.sqrt(2) * (new_data.seq - data.seq.mean())
    assert np.allclose(predicted[:, 0], 1)
    assert np.allclose(predicted[:, 1], expected)
    assert not np.isclose(predicted[:, 1].mean(), 0)
    # Predictions must not depend on the other observations in the prediction batch.
    single = dm.common.evaluate_new_data(new_data.iloc[:1]).design_matrix
    assert np.allclose(single, predicted[:1])


@pytest.mark.parametrize("intercept", [False, True])
def test_deprecated_intercept(sequence, intercept):
    spline = BSpline()
    with pytest.warns(FutureWarning, match="'intercept' is deprecated; use 'center' instead"):
        # The deprecated alias takes precedence over an explicitly conflicting center value.
        result = spline(sequence, df=4, intercept=intercept, center=intercept)
    expected = BSpline()(sequence, df=4, center=not intercept)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("kwargs", [{}, {"intercept": None}])
def test_bs_without_deprecated_intercept(sequence, kwargs, recwarn):
    matrix = BSpline()(sequence, df=3, **kwargs)
    assert np.allclose(matrix.mean(axis=0), 0)
    assert not recwarn


def test_deprecated_intercept_positional(sequence):
    with pytest.warns(FutureWarning, match="'intercept' is deprecated"):
        result = BSpline()(sequence, 4, None, 3, False, -1, 2)
    expected = BSpline()(sequence, df=4, lower_bound=-1, upper_bound=2, center=True)
    assert np.allclose(result, expected)


def test_uncentered_linear_basis(sequence):
    bs = BSpline()
    matrix = bs(sequence, 2, degree=1, center=False)
    true = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.9, 0.1],
            [0.85, 0.15],
            [0.8, 0.2],
            [0.75, 0.25],
            [0.7, 0.3],
            [0.65, 0.35],
            [0.6, 0.4],
            [0.55, 0.45],
            [0.5, 0.5],
            [0.45, 0.55],
            [0.4, 0.6],
            [0.35, 0.65],
            [0.3, 0.7],
            [0.25, 0.75],
            [0.2, 0.8],
            [0.15, 0.85],
            [0.1, 0.9],
            [0.05, 0.95],
            [0.0, 1.0],
        ]
    )
    assert np.allclose(matrix, true)


def test_invalid_degree(sequence):
    with pytest.raises(ValueError, match="'degree' must be an integer, not"):
        bs = BSpline()
        bs(sequence, degree=0.3)

    with pytest.raises(ValueError, match="'degree' must be greater than 0, not"):
        bs = BSpline()
        bs(sequence, degree=-1)


def test_df_and_knots_are_none(sequence):
    with pytest.raises(ValueError, match="Must specify either 'df' or 'knots'"):
        bs = BSpline()
        bs(sequence)


def test_invalid_df(sequence):
    with pytest.raises(ValueError, match="'df' must be either None or integer"):
        bs = BSpline()
        bs(sequence, df=[2])


def test_df_too_small_for_degree(sequence):
    with pytest.raises(
        ValueError, match="df=2 is too small for degree=3 and center=True; it must be >= 3"
    ):
        bs = BSpline()
        bs(sequence, df=2, degree=3)

    with pytest.raises(
        ValueError, match="df=2 is too small for degree=2 and center=False; it must be >= 3"
    ):
        bs = BSpline()
        bs(sequence, df=2, degree=2, center=False)


def test_provided_bad_number_of_knots(sequence):
    with pytest.raises(ValueError, match="df=5 with degree=3 implies 2 knots; but 3 were provided"):
        bs = BSpline()
        bs(sequence, df=5, degree=3, knots=[0.5, 0.6, 0.7])

    with pytest.raises(ValueError, match="df=5 with degree=3 implies 2 knots; but 1 were provided"):
        bs = BSpline()
        bs(sequence, df=5, degree=3, knots=[0.5])


def test_provided_bad_dimension_of_knots(sequence):
    with pytest.raises(ValueError, match="'knots' must be 1 dimensional"):
        bs = BSpline()
        bs(sequence, df=5, degree=3, knots=np.array([[0.5], [0.6]]))


def test_knots_dont_cover_range(sequence):
    with pytest.raises(
        ValueError, match=re.escape("Some knot values [1.2] fall above upper bound 1.0")
    ):
        bs = BSpline()
        bs(sequence, df=5, degree=3, knots=[0.3, 1.2])

    with pytest.raises(
        ValueError, match=re.escape("Some knot values [-0.1] fall below lower bound 0")
    ):
        bs = BSpline()
        bs(sequence, df=5, degree=3, knots=[-0.1, 0.3])
