import pytest
import re

import numpy as np
import pandas as pd

from formulae.matrices import design_matrices
from formulae.transforms import BSpline


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1234)
    size = 21
    data = pd.DataFrame(
        {"seq": np.linspace(0, 1, 21), "x1": rng.uniform(size=size), "x2": rng.uniform(size=size)}
    )
    return data


@pytest.fixture(scope="module")
def sequence():
    return np.linspace(0, 1, 21)


def test_basic(sequence):
    bs = BSpline()
    matrix = bs(sequence, 3)

    assert np.allclose(
        matrix[:, 0],
        [
            0.000000,
            0.135375,
            0.243000,
            0.325125,
            0.384000,
            0.421875,
            0.441000,
            0.443625,
            0.432000,
            0.408375,
            0.375000,
            0.334125,
            0.288000,
            0.238875,
            0.189000,
            0.140625,
            0.096000,
            0.057375,
            0.027000,
            0.007125,
            0.000000,
        ],
    )

    assert np.allclose(
        matrix[:, 1],
        [
            0.000000,
            0.007125,
            0.027000,
            0.057375,
            0.096000,
            0.140625,
            0.189000,
            0.238875,
            0.288000,
            0.334125,
            0.375000,
            0.408375,
            0.432000,
            0.443625,
            0.441000,
            0.421875,
            0.384000,
            0.325125,
            0.243000,
            0.135375,
            0.000000,
        ],
    )

    assert np.allclose(
        matrix[:, 2],
        [
            0.000000,
            0.000125,
            0.001000,
            0.003375,
            0.008000,
            0.015625,
            0.027000,
            0.042875,
            0.064000,
            0.091125,
            0.125000,
            0.166375,
            0.216000,
            0.274625,
            0.343000,
            0.421875,
            0.512000,
            0.614125,
            0.729000,
            0.857375,
            1.000000,
        ],
    )


def test_degree(sequence):
    bs = BSpline()
    matrix = bs(sequence, df=1, degree=1)
    true = np.array(
        [
            [
                0.0,
                0.05,
                0.1,
                0.15,
                0.2,
                0.25,
                0.3,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]
        ]
    )
    assert np.allclose(matrix, true.T)


def test_basic_through_design_matrices(data):
    dm = design_matrices("bs(seq, 3) - 1", data)
    matrix = dm.common.design_matrix
    true = np.array(
        [
            [
                0.000000,
                0.135375,
                0.243000,
                0.325125,
                0.384000,
                0.421875,
                0.441000,
                0.443625,
                0.432000,
                0.408375,
                0.375000,
                0.334125,
                0.288000,
                0.238875,
                0.189000,
                0.140625,
                0.096000,
                0.057375,
                0.027000,
                0.007125,
                0.000000,
            ],
            [
                0.000000,
                0.007125,
                0.027000,
                0.057375,
                0.096000,
                0.140625,
                0.189000,
                0.238875,
                0.288000,
                0.334125,
                0.375000,
                0.408375,
                0.432000,
                0.443625,
                0.441000,
                0.421875,
                0.384000,
                0.325125,
                0.243000,
                0.135375,
                0.000000,
            ],
            [
                0.000000,
                0.000125,
                0.001000,
                0.003375,
                0.008000,
                0.015625,
                0.027000,
                0.042875,
                0.064000,
                0.091125,
                0.125000,
                0.166375,
                0.216000,
                0.274625,
                0.343000,
                0.421875,
                0.512000,
                0.614125,
                0.729000,
                0.857375,
                1.000000,
            ],
        ]
    )

    assert np.allclose(matrix, true.T)
    labels = dm.common.terms["bs(seq, 3)"].labels
    assert labels == ["bs(seq, 3)[0]", "bs(seq, 3)[1]", "bs(seq, 3)[2]"]


def test_intercept(sequence):
    bs = BSpline()
    matrix = bs(sequence, 2, degree=1, intercept=True)
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
        ValueError, match="df=2 is too small for degree=3 and intercept=False; it must be >= 3"
    ):
        bs = BSpline()
        bs(sequence, df=2, degree=3)

    with pytest.raises(
        ValueError, match="df=2 is too small for degree=2 and intercept=True; it must be >= 3"
    ):
        bs = BSpline()
        bs(sequence, df=2, degree=2, intercept=True)


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
