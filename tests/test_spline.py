import pytest
import re

import numpy as np
import pandas as pd

from formulae.matrices import design_matrices
from formulae.transforms import (
    BSpline,
    CyclicCubicSpline,
    NaturalCubicSpline,
    _CubicRegressionSpline,
)


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


class TestCyclicCubicSpline:
    @pytest.fixture
    def hours(self):
        return np.arange(24, dtype=float)

    def test_shape_and_centering(self, hours):
        spline = CyclicCubicSpline()
        basis = spline(hours, period=24, df=6)

        assert basis.shape == (24, 6)
        assert np.allclose(basis.mean(axis=0), 0)
        assert spline.period == 24
        assert spline.bounds == (0, 24)

    def test_default_df(self, hours):
        assert CyclicCubicSpline()(hours, period=24).shape == (24, 10)

    def test_mgcv_ground_truth(self):
        # x was generated in Python and copied verbatim to mgcv 1.9-1:
        # knots <- c(0, 1.23829311, 2.35619188, 6.22059918, 6.75623046, 12)
        # smoothCon(s(x, bs="cc", k=6), data.frame(x=x),
        #           knots=list(x=knots), absorb.cons=TRUE)[[1]]$X
        x = np.array(
            [
                1.06051805,
                1.23829311,
                0.85024307,
                6.47992182,
                0.21034573,
                2.15650251,
                6.38249844,
                9.3260484,
                2.35619188,
                4.55642545,
                8.43365906,
                1.68455449,
                4.98185406,
                6.75623046,
                6.22059918,
                8.1296006,
            ]
        )
        expected = np.array(
            [
                [0.934586924483147, -0.175754715230952, -0.013728215297758, -0.182253139436015],
                [0.966439924978457, -0.070078707710494, -0.026690191275200, -0.115167220101333],
                [0.802403306916357, -0.271116999321654, -0.011771324482471, -0.290137587906711],
                [-0.053840772108119, -0.152760120088598, 0.474528382972572, 0.250018269333851],
                [0.050225002605978, -0.425837984090704, -0.105482974968229, -0.660001794011894],
                [0.156631389259803, 0.705113196943741, -0.095970612903064, -0.171252007888193],
                [-0.039605602480673, -0.118264800810937, 0.673273840340725, 0.118402755863938],
                [-1.117102311204004, -0.619744434706695, -1.665935836860772, 0.524427874157157],
                [-0.070078707710494, 0.853664651487803, -0.055733311439599, -0.240487244147420],
                [-0.452807231349719, 0.798422602262183, 1.631276792847306, -0.967210070202182],
                [-0.787152154738540, -0.565805447182774, -1.754488101639260, 1.064665385382963],
                [0.705011117223790, 0.289747240101040, -0.084677743556385, -0.071616624693189],
                [-0.300219005053078, 0.578836120082767, 1.806659171014440, -0.932909334754151],
                [-0.115167220101333, -0.240487244147420, -0.091592022102589, 0.604783702737410],
                [-0.026690191275200, -0.055733311439599, 0.978773399348794, -0.091592022102589],
                [-0.652634469446372, -0.530200046147707, -1.658441251998509, 1.160329057768357],
            ]
        )

        spline = CyclicCubicSpline()
        result = spline(x, period=12, df=4, lower_bound=0)
        assert np.allclose(result, expected)
        assert np.allclose(
            spline._knots,
            [0, 1.23829311, 2.35619188, 6.22059918, 6.75623046, 12],
        )

    def test_periodic_boundary_conditions(self, hours):
        spline = CyclicCubicSpline()
        spline(hours, period=24, df=6, center=False)

        for derivative in range(3):
            left = spline._spline(0, nu=derivative)  # pylint: disable=protected-access
            right = spline._spline(24, nu=derivative)  # pylint: disable=protected-access
            assert np.allclose(left, right)

    def test_wrapping(self, hours):
        spline = CyclicCubicSpline()
        spline(hours, period=24, df=6)

        reference = spline.eval(np.array([0, 3, 12, 23], dtype=float))
        wrapped = spline.eval(np.array([24, 27, -12, 47], dtype=float))
        assert np.allclose(reference, wrapped)

    def test_multiple_observed_periods(self):
        hours = np.arange(7 * 24, dtype=float)
        spline = CyclicCubicSpline()
        basis = spline(hours, period=24, df=6)

        assert spline.bounds == (0, 24)
        for day in range(1, 7):
            assert np.allclose(basis[:24], basis[day * 24 : (day + 1) * 24])

    def test_quantile_knots_use_unique_wrapped_phases(self):
        phases_over_two_periods = np.array([0, 1, 4, 10, 12, 13, 16, 22], dtype=float)
        spline = CyclicCubicSpline()
        spline(phases_over_two_periods, period=12, df=3, lower_bound=0)

        assert np.allclose(spline._knots, [0, 0.75, 2.5, 5.5, 12])

    def test_through_design_matrices_and_new_data(self, hours):
        data = pd.DataFrame({"hour": hours})
        dm = design_matrices("cc(hour, period=24, df=5) - 1", data)
        new_data = pd.DataFrame({"hour": [-24, 0, 24, 48, 6]})
        new_dm = dm.common.evaluate_new_data(new_data)

        assert dm.common.design_matrix.shape == (24, 5)
        assert np.allclose(new_dm.design_matrix[0], new_dm.design_matrix[1])
        assert np.allclose(new_dm.design_matrix[1], new_dm.design_matrix[2])
        assert np.allclose(new_dm.design_matrix[2], new_dm.design_matrix[3])

        transform = (
            dm.common.terms["cc(hour, period=24, df=5)"].components[0].call.stateful_transform
        )
        new_transform = (
            new_dm.terms["cc(hour, period=24, df=5)"].components[0].call.stateful_transform
        )
        assert np.array_equal(transform._knots, new_transform._knots)

    def test_explicit_lower_bound_and_knots(self, hours):
        spline = CyclicCubicSpline()
        basis = spline(
            hours,
            period=24,
            knots=[6, 12, 18],
            lower_bound=0,
            center=True,
        )

        assert basis.shape == (24, 3)
        assert np.array_equal(spline._knots, [0, 6, 12, 18, 24])

    @pytest.mark.parametrize("period", [0, -24, np.inf, np.nan, "day"])
    def test_invalid_period(self, hours, period):
        with pytest.raises(ValueError, match="'period'"):
            CyclicCubicSpline()(hours, period=period, df=5)

    @pytest.mark.parametrize(
        "knots, match",
        [
            ([[6], [12]], "one dimensional"),
            ([6, 6], "repeated"),
            ([0, 12], "strictly between"),
            ([6, 24], "strictly between"),
            ([6, np.nan], "finite"),
        ],
    )
    def test_invalid_knots(self, hours, knots, match):
        with pytest.raises(ValueError, match=match):
            CyclicCubicSpline()(
                hours,
                period=24,
                df=None,
                knots=knots,
                lower_bound=0,
            )

    def test_df_and_knots_must_agree(self, hours):
        with pytest.raises(ValueError, match="implies 5 knots, but 3 were provided"):
            CyclicCubicSpline()(
                hours,
                period=24,
                df=5,
                knots=[6, 12, 18],
                lower_bound=0,
                center=True,
            )


class TestNaturalCubicSpline:
    @pytest.fixture
    def sequence(self):
        return np.linspace(0, 10, 21)

    def test_shape_and_centering(self, sequence):
        spline = NaturalCubicSpline()
        basis = spline(sequence, df=5)

        assert basis.shape == (21, 5)
        assert np.allclose(basis.mean(axis=0), 0)
        assert spline.bounds == (0, 10)

    def test_mgcv_ground_truth(self):
        # x was generated in Python and copied verbatim to mgcv 1.9-1:
        # smoothCon(s(x, bs="cr", k=5), data.frame(x=x),
        #           absorb.cons=TRUE)[[1]]$X
        x = np.array(
            [
                7.74031445,
                1.49492503,
                0.58818596,
                9.50374841,
                2.5347688,
                7.70885074,
                8.66345652,
                4.64862233,
                6.01866163,
                6.46400334,
                5.89970973,
                4.66060134,
                0.29105762,
            ]
        )
        expected = np.array(
            [
                [-0.132736520848244, -0.225101757429345, 0.872139532842461, -0.065171196464005],
                [0.456197055762551, -0.504419090804647, -0.179431979367755, -0.141680724823798],
                [-0.194991214248821, -0.648175048694668, -0.351102174222237, -0.212985910061789],
                [-0.077259079896347, -0.122667885120055, -0.074670551517556, 0.957581990740286],
                [0.859282282912341, -0.223424156451565, -0.136003037537128, -0.077259079896347],
                [-0.136003037537128, -0.215938437358634, 0.868553678938643, -0.074670551517556],
                [-0.080812341324650, -0.281945940090955, 0.582323083602034, 0.382298051584501],
                [0.236464674721593, 0.625100964773260, -0.482056967423693, -0.052171108384081],
                [-0.241474307361239, 0.606211864918110, -0.147038512349720, -0.133096843777951],
                [-0.263269795975454, 0.403909367736128, 0.164049276190487, -0.170078710119635],
                [-0.223424156451565, 0.645258928873916, -0.215938437358634, -0.122667885120055],
                [0.230750154574136, 0.628247040546821, -0.482598397761715, -0.052517173756211],
                [-0.432723714327174, -0.687055850898366, -0.418225514035189, -0.237580858403360],
            ]
        )

        spline = NaturalCubicSpline()
        result = spline(x, df=4)
        assert np.allclose(result, expected)
        assert np.allclose(
            spline._knots,
            [0.29105762, 2.5347688, 5.89970973, 7.70885074, 9.50374841],
        )

    def test_natural_boundary_conditions(self, sequence):
        spline = NaturalCubicSpline()
        spline(sequence, df=5, center=False)

        assert np.allclose(spline._spline(0, nu=2), 0, atol=1e-12)
        assert np.allclose(spline._spline(10, nu=2), 0, atol=1e-12)

    def test_linear_extrapolation(self, sequence):
        spline = NaturalCubicSpline()
        spline(sequence, df=5)

        left = spline.eval(np.array([-4, -2, 0], dtype=float))
        right = spline.eval(np.array([10, 12, 14], dtype=float))

        # Equally spaced points on each tail have zero second finite difference.
        assert np.allclose(left[0] - 2 * left[1] + left[2], 0)
        assert np.allclose(right[0] - 2 * right[1] + right[2], 0)

        centering_matrix = spline._centering_matrix
        left_slope = spline._spline(0, nu=1) @ centering_matrix
        right_slope = spline._spline(10, nu=1) @ centering_matrix
        assert np.allclose((left[2] - left[1]) / 2, left_slope)
        assert np.allclose((right[1] - right[0]) / 2, right_slope)

    def test_through_design_matrices_and_new_data(self, sequence):
        data = pd.DataFrame({"x": sequence})
        dm = design_matrices("cr(x, df=4) - 1", data)
        new_data = pd.DataFrame({"x": [-4, -2, 0, 10, 12, 14]})
        new_dm = dm.common.evaluate_new_data(new_data)
        matrix = new_dm.design_matrix

        assert dm.common.design_matrix.shape == (21, 4)
        assert np.allclose(matrix[0] - 2 * matrix[1] + matrix[2], 0)
        assert np.allclose(matrix[3] - 2 * matrix[4] + matrix[5], 0)

    def test_quantile_knots_use_unique_values(self):
        x = np.array([0, 0, 1, 4, 10, 10], dtype=float)
        spline = NaturalCubicSpline()
        spline(x, df=3)

        assert np.allclose(spline._knots, [0, 1, 4, 10])

    def test_bounds_are_added_after_computing_quantiles(self):
        x = np.array([2, 3, 4, 5], dtype=float)
        spline = NaturalCubicSpline()
        spline(x, df=3, lower_bound=0, upper_bound=10)

        assert np.allclose(spline._knots, [0, 3, 4, 10])

    def test_explicit_knots(self, sequence):
        spline = NaturalCubicSpline()
        basis = spline(sequence, knots=[2.5, 5, 7.5], lower_bound=0, upper_bound=10)

        assert basis.shape == (21, 4)
        assert np.array_equal(spline._knots, [0, 2.5, 5, 7.5, 10])

    def test_unconstrained_basis(self, sequence):
        spline = NaturalCubicSpline()
        basis = spline(sequence, df=5, center=False)

        assert basis.shape == (21, 5)
        assert np.allclose(basis.sum(axis=1), 1)

    def test_invalid_bounds(self, sequence):
        with pytest.raises(ValueError, match="less than"):
            NaturalCubicSpline()(sequence, df=5, lower_bound=10, upper_bound=0)

    @pytest.mark.parametrize("df, center", [(0, True), (1, False)])
    def test_df_too_small(self, sequence, df, center):
        with pytest.raises(ValueError, match="greater than or equal"):
            NaturalCubicSpline()(sequence, df=df, center=center)


@pytest.mark.parametrize(
    "formula, term_name",
    [
        ("cr(x, df=4)", "cr(x, df=4)"),
        ("cc(x, period=12, df=4)", "cc(x, period=12, df=4)"),
    ],
)
def test_centered_spline_with_model_intercept(formula, term_name):
    data = pd.DataFrame({"x": np.linspace(0, 11, 24)})
    dm = design_matrices(formula, data)
    matrix = dm.common.design_matrix
    smooth = dm.common.terms[term_name].data

    assert matrix.shape == (24, 5)
    assert np.allclose(matrix[:, 0], 1)
    assert np.allclose(smooth.mean(axis=0), 0)
    assert np.linalg.matrix_rank(matrix) == 5


@pytest.mark.parametrize(
    "formula",
    [
        "cr(x, df=4, center=False) - 1",
        "cc(x, period=12, df=4, center=False) - 1",
    ],
)
def test_uncentered_spline_without_model_intercept(formula):
    data = pd.DataFrame({"x": np.linspace(0, 11, 24)})
    matrix = design_matrices(formula, data).common.design_matrix

    assert matrix.shape == (24, 4)
    assert np.allclose(matrix.sum(axis=1), 1)
    assert np.linalg.matrix_rank(matrix) == 4
