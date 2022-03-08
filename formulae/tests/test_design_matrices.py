from multiprocessing.sharedctypes import Value
import pytest
import re

import numpy as np
import pandas as pd

from formulae.matrices import (
    design_matrices,
    ResponseVector,
    CommonEffectsMatrix,
    GroupEffectsMatrix,
)
from formulae.parser import ParseError
from formulae.resolver import ResolverError


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1234)
    size = 20
    data = pd.DataFrame(
        {
            "y": rng.uniform(size=size),
            "x1": rng.uniform(size=size),
            "x2": rng.uniform(size=size),
            "x3": [1, 2, 3, 4] * 5,
            "f": rng.choice(["A", "B"], size=size),
            "g": rng.choice(["A", "B"], size=size),
            "h": rng.choice(["A", "B"], size=size),
            "j": rng.choice(["A", "B"], size=size),
        }
    )

    data["g"] = pd.Categorical(data["g"], categories=["A", "B"], ordered=True)
    return data


@pytest.fixture(scope="module")
def pixel():
    """
    X-ray pixel intensities over time dataset from R nlme package.
    The output is a subset of this dataframe.
    """
    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data")
    data = pd.read_csv(join(data_dir, "Pixel.csv"))

    data["Dog"] = data["Dog"].astype("category")
    data["day"] = data["day"].astype("category")
    data = data[data["Dog"].isin([1, 2, 3])]
    data = data[data["day"].isin([2, 4, 6])]
    data = data.sort_values(["Dog", "Side", "day"])
    data = data.reset_index(drop=True)
    return data


@pytest.fixture(scope="module")
def beetle():
    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )
    return data


def compare_dicts(d1, d2):
    if len(d1) != len(d2):
        return False
    if set(d1.keys()) != set(d2.keys()):
        return False

    for key in d1.keys():
        if type(d1[key]) != type(d2[key]):
            return False
        elif isinstance(d1[key], dict):
            outcome = compare_dicts(d1[key], d2[key])
            if not outcome:
                return False
        elif isinstance(d1[key], np.ndarray):
            if not all(d1[key] == d2[key]):
                return False
        else:
            if d1[key] != d2[key]:
                return False
    return True


def test_empty_formula(data):
    with pytest.raises(ValueError):
        design_matrices("", data)


def test_empty_model(data):
    dm = design_matrices("y ~ 0", data)
    assert dm.common == None
    assert dm.group == None


def test_common_intercept_only_model(data):
    dm = design_matrices("y ~ 1", data)
    assert len(dm.common.terms) == 1
    assert dm.common.terms["Intercept"].kind == "intercept"
    assert dm.common.terms["Intercept"].labels == ["Intercept"]
    assert all(dm.common.design_matrix == 1)
    assert dm.group == None


def test_group_specific_intercept_only(data):
    dm = design_matrices("y ~ 0 + (1|g)", data)
    assert len(dm.group.terms) == 1
    assert dm.group.terms["1|g"].kind == "intercept"
    assert dm.group.terms["1|g"].groups == ["A", "B"]
    assert dm.group.terms["1|g"].labels == ["1|g[A]", "1|g[B]"]
    assert dm.common == None


def test_common_predictor(data):
    dm = design_matrices("y ~ x1", data)
    assert list(dm.common.terms) == ["Intercept", "x1"]
    assert dm.common.terms["x1"].kind == "numeric"
    assert dm.common.terms["x1"].labels == ["x1"]
    assert dm.common.terms["x1"].levels is None

    # 'f' does not span intercept because the intercept is already icluded
    dm = design_matrices("y ~ f", data)
    assert list(dm.common.terms) == ["Intercept", "f"]
    assert dm.common.terms["f"].kind == "categoric"
    assert dm.common.terms["f"].labels == [f"f[{l}]" for l in sorted(data["f"].unique())[1:]]
    assert dm.common.terms["f"].levels == sorted(list(data["f"].unique()))[1:]
    assert dm.common.terms["f"].spans_intercept == False


def test_categoric_encoding(data):
    # No intercept, one categoric predictor
    dm = design_matrices("y ~ 0 + f", data)
    assert list(dm.common.terms) == ["f"]
    assert dm.common.terms["f"].kind == "categoric"
    assert dm.common.terms["f"].labels == [f"f[{l}]" for l in sorted(data["f"].unique())]
    assert dm.common.terms["f"].levels == sorted(list(data["f"].unique()))
    assert dm.common.terms["f"].spans_intercept is True
    assert dm.common.design_matrix.shape == (20, 2)

    # Intercept, one categoric predictor
    dm = design_matrices("y ~ 1 + f", data)
    assert list(dm.common.terms) == ["Intercept", "f"]
    assert dm.common.terms["f"].kind == "categoric"
    assert dm.common.terms["f"].labels == [f"f[{l}]" for l in sorted(data["f"].unique())[1:]]
    assert dm.common.terms["f"].levels == sorted(list(data["f"].unique()))[1:]
    assert dm.common.terms["f"].spans_intercept is False

    assert dm.common.design_matrix.shape == (20, 2)

    # No intercept, two additive categoric predictors
    dm = design_matrices("y ~ 0 + f + g", data)
    assert list(dm.common.terms) == ["f", "g"]
    assert dm.common.terms["f"].kind == "categoric"
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["f"].labels == [f"f[{l}]" for l in sorted(data["f"].unique())]
    assert dm.common.terms["g"].labels == [f"g[{l}]" for l in sorted(data["g"].unique())[1:]]
    assert dm.common.terms["f"].levels == sorted(list(data["f"].unique()))
    assert dm.common.terms["g"].levels == sorted(list(data["g"].unique()))[1:]
    assert dm.common.terms["f"].spans_intercept is True
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.design_matrix.shape == (20, 3)

    # Intercept, two additive categoric predictors
    dm = design_matrices("y ~ 1 + f + g", data)
    assert list(dm.common.terms) == ["Intercept", "f", "g"]
    assert dm.common.terms["f"].kind == "categoric"
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["f"].labels == [f"f[{l}]" for l in sorted(data["f"].unique())[1:]]
    assert dm.common.terms["g"].labels == [f"g[{l}]" for l in sorted(data["g"].unique())[1:]]
    assert dm.common.terms["f"].levels == sorted(list(data["f"].unique()))[1:]
    assert dm.common.terms["g"].levels == sorted(list(data["g"].unique()))[1:]
    assert dm.common.terms["f"].spans_intercept is False
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.design_matrix.shape == (20, 3)

    # No intercept, two categoric predictors with interaction
    dm = design_matrices("y ~ 0 + f + g + f:g", data)
    assert list(dm.common.terms) == ["f", "g", "f:g"]
    assert dm.common.terms["f"].kind == "categoric"
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["f:g"].kind == "interaction"
    assert dm.common.terms["f"].labels == [f"f[{l}]" for l in sorted(data["f"].unique())]
    assert dm.common.terms["g"].labels == [f"g[{l}]" for l in sorted(data["g"].unique())[1:]]
    assert dm.common.terms["f:g"].labels == ["f[B]:g[B]"]
    assert dm.common.terms["f"].levels == sorted(list(data["f"].unique()))
    assert dm.common.terms["g"].levels == sorted(list(data["g"].unique()))[1:]
    assert dm.common.terms["f:g"].levels == ["B, B"]
    assert dm.common.terms["f"].spans_intercept is True
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["f:g"].spans_intercept is False
    assert dm.common.terms["f:g"].components[0].spans_intercept is False
    assert dm.common.terms["f:g"].components[1].spans_intercept is False
    assert dm.common.design_matrix.shape == (20, 4)

    # Intercept, two categoric predictors with interaction
    dm = design_matrices("y ~ 1 + f + g + f:g", data)
    assert list(dm.common.terms) == ["Intercept", "f", "g", "f:g"]
    assert dm.common.terms["f"].kind == "categoric"
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["f:g"].kind == "interaction"
    assert dm.common.terms["f"].labels == [f"f[{l}]" for l in sorted(data["f"].unique())[1:]]
    assert dm.common.terms["g"].labels == [f"g[{l}]" for l in sorted(data["g"].unique())[1:]]
    assert dm.common.terms["f:g"].labels == ["f[B]:g[B]"]
    assert dm.common.terms["f"].levels == sorted(list(data["f"].unique()))[1:]
    assert dm.common.terms["g"].levels == sorted(list(data["g"].unique()))[1:]
    assert dm.common.terms["f"].spans_intercept is False
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["f:g"].components[0].spans_intercept is False
    assert dm.common.terms["f:g"].components[1].spans_intercept is False
    assert dm.common.design_matrix.shape == (20, 4)

    # No intercept, interaction between two categorics
    dm = design_matrices("y ~ 0 + f:g", data)
    assert list(dm.common.terms) == ["f:g"]
    assert dm.common.terms["f:g"].kind == "interaction"
    assert dm.common.terms["f:g"].labels == ["f[A]:g[A]", "f[A]:g[B]", "f[B]:g[A]", "f[B]:g[B]"]
    assert dm.common.terms["f:g"].spans_intercept is True
    assert dm.common.terms["f:g"].components[0].spans_intercept is True
    assert dm.common.terms["f:g"].components[1].spans_intercept is True
    assert dm.common.design_matrix.shape == (20, 4)

    # Intercept, interaction between two categorics
    # It adds "g" -> It uses Patsy algorithm..
    dm = design_matrices("y ~ 1 + f:g", data)
    assert list(dm.common.terms) == ["Intercept", "g", "f:g"]
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["f:g"].kind == "interaction"
    assert dm.common.terms["f:g"].labels == ["f[B]:g[A]", "f[B]:g[B]"]
    assert dm.common.terms["f:g"].spans_intercept is False
    assert dm.common.terms["f:g"].components[0].spans_intercept is False
    assert dm.common.terms["f:g"].components[1].spans_intercept is True
    assert dm.common.design_matrix.shape == (20, 4)

    # Same than before
    dm = design_matrices("y ~ 1 + g + f:g", data)
    assert list(dm.common.terms) == ["Intercept", "g", "f:g"]
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["f:g"].kind == "interaction"
    assert dm.common.terms["f:g"].labels == ["f[B]:g[A]", "f[B]:g[B]"]
    assert dm.common.terms["f:g"].spans_intercept is False
    assert dm.common.terms["f:g"].components[0].spans_intercept is False
    assert dm.common.terms["f:g"].components[1].spans_intercept is True
    assert dm.common.design_matrix.shape == (20, 4)


def test_categoric_encoding_with_numeric_interaction(data):
    dm = design_matrices("y ~ x1 + x2 + f:g + h:j:x2", data)
    assert list(dm.common.terms) == ["Intercept", "x1", "x2", "g", "f:g", "j:x2", "h:j:x2"]
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["f:g"].kind == "interaction"
    assert dm.common.terms["f:g"].labels == ["f[B]:g[A]", "f[B]:g[B]"]
    assert dm.common.terms["f:g"].components[0].spans_intercept is False
    assert dm.common.terms["f:g"].components[1].spans_intercept is True
    assert dm.common.terms["j:x2"].spans_intercept is False
    assert dm.common.terms["h:j:x2"].components[0].spans_intercept is False
    assert dm.common.terms["h:j:x2"].components[1].spans_intercept is True
    assert dm.common.terms["h:j:x2"].components[2].kind == "numeric"


def test_interactions(data):
    # These two models are the same
    dm = design_matrices("y ~ f * g", data)
    dm2 = design_matrices("y ~ f + g + f:g", data)
    assert dm2.common.terms == dm.common.terms

    # When no intercept too
    dm = design_matrices("y ~ 0 + f * g", data)
    dm2 = design_matrices("y ~ 0 + f + g + f:g", data)
    assert dm2.common.terms == dm.common.terms

    # Mix of numeric/categoric
    # "g" in "g" -> does not span intercept
    # "g" in "x1:g" -> does not span intercept because x1 is present in formula
    dm = design_matrices("y ~ x1 + g + x1:g", data)
    assert list(dm.common.terms) == ["Intercept", "x1", "g", "x1:g"]
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["x1:g"].components[1].spans_intercept is False

    # "g" in "g" -> reduced
    # "g" in "x1:g" -> full because x1 is not present in formula
    dm = design_matrices("y ~ g + x1:g", data)
    assert list(dm.common.terms) == ["Intercept", "g", "x1:g"]
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["x1:g"].components[1].spans_intercept is True

    # "g" in "x1:x2:g" is full, because x1:x2 is a new group and we don't have x1:x2 in the model
    dm = design_matrices("y ~ x1 + g + x1:g + x1:x2:g", data)
    assert list(dm.common.terms) == ["Intercept", "x1", "g", "x1:g", "x1:x2:g"]
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["x1:g"].components[1].spans_intercept is False
    assert dm.common.terms["x1:x2:g"].components[2].spans_intercept is True

    # "g" in "x1:x2:g" is reduced, because x1:x2 is a new group and we have x1:x2 in the model
    dm = design_matrices("y ~ x1 + g + x1:x2 + x1:g + x1:x2:g", data)
    assert list(dm.common.terms) == ["Intercept", "x1", "g", "x1:x2", "x1:g", "x1:x2:g"]
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["g"].spans_intercept is False
    assert dm.common.terms["x1:g"].components[1].spans_intercept is False
    assert dm.common.terms["x1:x2:g"].components[2].spans_intercept is False

    # And now, since we don't have intercept, x1 and x1:x2 all "g" are full
    dm = design_matrices("y ~ 0 + g + x1:g + x1:x2:g", data)
    assert list(dm.common.terms) == ["g", "x1:g", "x1:x2:g"]
    assert dm.common.terms["g"].kind == "categoric"
    assert dm.common.terms["g"].spans_intercept is True
    assert dm.common.terms["x1:g"].components[1].spans_intercept is True
    assert dm.common.terms["x1:x2:g"].components[2].spans_intercept is True

    # Two numerics
    dm = design_matrices("y ~ x1:x2", data)
    assert "x1:x2" in dm.common.terms
    assert np.allclose(dm.common["x1:x2"][:, 0], data["x1"] * data["x2"])


def test_built_in_transforms(data):
    # {...} gets translated to I(...)
    dm = design_matrices("y ~ {x1 + x2}", data)
    assert list(dm.common.terms) == ["Intercept", "I(x1 + x2)"]
    assert dm.common.terms["I(x1 + x2)"].kind == "numeric"
    assert np.allclose(dm.common["I(x1 + x2)"], (data["x1"] + data["x2"]).values[:, None])

    dm2 = design_matrices("y ~ I(x1 + x2)", data)
    assert compare_dicts(dm.common.terms, dm2.common.terms)

    # center()
    dm = design_matrices("y ~ center(x1)", data)
    assert list(dm.common.terms) == ["Intercept", "center(x1)"]
    assert dm.common.terms["center(x1)"].kind == "numeric"
    assert np.allclose(dm.common["center(x1)"].mean(), 0)

    # scale()
    dm = design_matrices("y ~ scale(x1)", data)
    assert list(dm.common.terms) == ["Intercept", "scale(x1)"]
    assert dm.common.terms["scale(x1)"].kind == "numeric"
    assert np.allclose(dm.common["scale(x1)"].mean(), 0)
    assert np.allclose(dm.common["scale(x1)"].std(), 1)

    # standardize(), alias of scale()
    dm = design_matrices("y ~ standardize(x1)", data)
    assert list(dm.common.terms) == ["Intercept", "standardize(x1)"]
    assert dm.common.terms["standardize(x1)"].kind == "numeric"
    assert np.allclose(dm.common["standardize(x1)"].mean(), 0)
    assert np.allclose(dm.common["standardize(x1)"].std(), 1)

    # C()
    # Intercept, no extra arguments, reference is first value observed
    dm = design_matrices("y ~ C(x3)", data)
    assert list(dm.common.terms) == ["Intercept", "C(x3)"]
    assert dm.common.terms["C(x3)"].kind == "categoric"
    assert dm.common.terms["C(x3)"].spans_intercept is False
    assert dm.common.terms["C(x3)"].levels == ["2", "3", "4"]
    assert dm.common.terms["C(x3)"].labels == ["C(x3)[2]", "C(x3)[3]", "C(x3)[4]"]

    # No intercept, no extra arguments
    dm = design_matrices("y ~ 0 + C(x3)", data)
    assert list(dm.common.terms) == ["C(x3)"]
    assert dm.common.terms["C(x3)"].kind == "categoric"
    assert dm.common.terms["C(x3)"].spans_intercept is True
    assert dm.common.terms["C(x3)"].levels == ["1", "2", "3", "4"]
    assert dm.common.terms["C(x3)"].labels == ["C(x3)[1]", "C(x3)[2]", "C(x3)[3]", "C(x3)[4]"]

    # Specify levels, different to observed
    lvls = [3, 2, 4, 1]  # noqa
    dm = design_matrices("y ~ C(x3, levels=lvls)", data)
    assert dm.common.terms["C(x3, levels = lvls)"].kind == "categoric"
    assert dm.common.terms["C(x3, levels = lvls)"].levels == ["2", "4", "1"]

    # Pass a reference not in the data
    with pytest.raises(ValueError):
        dm = design_matrices("y ~ C(x3, 5)", data)

    # Pass categoric, remains unchanged
    dm = design_matrices("y ~ C(f)", data)
    dm2 = design_matrices("y ~ f", data)
    d1 = dm.common.terms["C(f)"]
    d2 = dm2.common.terms["f"]
    assert d1.kind == d2.kind
    assert d1.levels == d2.levels

    assert d1.spans_intercept == d2.spans_intercept
    assert not d1.labels == d2.labels  # because one is 'C(f)' and other is 'f'
    assert all(dm.common["C(f)"] == dm2.common["f"])


def test_external_transforms(data):
    dm = design_matrices("y ~ np.exp(x1)", data)
    assert np.allclose(dm.common["np.exp(x1)"][:, 0], np.exp(data["x1"]))

    def add_ten(x):
        return x + 10

    dm = design_matrices("y ~ add_ten(x1)", data)
    assert np.allclose(dm.common["add_ten(x1)"][:, 0], data["x1"] + 10)


def test_non_syntactic_names():
    data = pd.DataFrame(
        {
            "My response": np.random.normal(size=10),
            "$$#1@@": np.random.normal(size=10),
            "-- ! Hi there!": np.random.normal(size=10),
        }
    )

    dm = design_matrices("`My response` ~ `$$#1@@`*`-- ! Hi there!`", data)
    assert list(dm.common.terms.keys()) == [
        "Intercept",
        "$$#1@@",
        "-- ! Hi there!",
        "$$#1@@:-- ! Hi there!",
    ]
    assert np.allclose(dm.common["$$#1@@"][:, 0], data["$$#1@@"])
    assert np.allclose(dm.common["-- ! Hi there!"][:, 0], data["-- ! Hi there!"])
    assert np.allclose(dm.common["-- ! Hi there!"][:, 0], data["-- ! Hi there!"])
    assert np.allclose(
        dm.common["$$#1@@:-- ! Hi there!"][:, 0], data["$$#1@@"] * data["-- ! Hi there!"]
    )


def test_categoric_group_specific():
    data = pd.DataFrame(
        {
            "BP": np.random.normal(size=30),
            "BMI": np.random.normal(size=30),
            "age_grp": np.random.choice([0, 1, 2], size=30),
        }
    )
    dm = design_matrices("BP ~ 0 + (C(age_grp)|BMI)", data)
    list(dm.group.terms.keys()) == ["1|BMI", "C(age_grp)[1]|BMI", "C(age_grp)[2]|BMI"]

    dm = design_matrices("BP ~ 0 + (0 + C(age_grp)|BMI)", data)
    list(dm.group.terms) == ["C(age_grp)[0]|BMI", "C(age_grp)[1]|BMI", "C(age_grp)[2]|BMI"]


def test_interactions_in_group_specific(pixel):
    # We have group specific terms with the following characteristics
    # 1. expr=categoric, factor=categoric
    # 2. expr=intercept, factor=categoric
    # 3. expr=intercept, factor=interaction between categorics
    # The desing matrices used for the comparison are loaded from text files.
    # The encoding is implicitly checked when comparing names.

    from os.path import dirname, join

    data_dir = join(dirname(__file__), "data/group_specific")
    slope_by_dog_original = np.loadtxt(join(data_dir, "slope_by_dog.txt"))
    intercept_by_side_original = np.loadtxt(join(data_dir, "intercept_by_side.txt"))
    intercept_by_side_dog_original = np.loadtxt(join(data_dir, "intercept_by_side_dog.txt"))
    dog_and_side_by_day_original = np.loadtxt(join(data_dir, "dog_and_side_by_day.txt"))

    dm = design_matrices("pixel ~ day +  (0 + day | Dog) + (1 | Side/Dog)", pixel)
    slope_by_dog = dm.group["day|Dog"]
    intercept_by_side = dm.group["1|Side"]
    intercept_by_side_dog = dm.group["1|Side:Dog"]

    # Assert values in the design matrix
    assert (slope_by_dog == slope_by_dog_original).all()
    assert (intercept_by_side == intercept_by_side_original).all()
    assert (intercept_by_side_dog == intercept_by_side_dog_original).all()

    # Assert labels
    names = [f"day[{d}]|Dog[{g}]" for g in [1, 2, 3] for d in [2, 4, 6]]
    assert dm.group.terms["day|Dog"].labels == names
    names = [f"1|Side[{s}]" for s in ["L", "R"]]
    assert dm.group.terms["1|Side"].labels == names
    names = [f"1|Side[{s}]:Dog[{d}]" for s in ["L", "R"] for d in [1, 2, 3]]
    assert dm.group.terms["1|Side:Dog"].labels == names

    # Another design matrix
    dm = design_matrices("(0 + Dog:Side | day)", pixel)
    dog_and_side_by_day = dm.group["Dog:Side|day"]

    # Assert values in the design matrix
    assert (dog_and_side_by_day == dog_and_side_by_day_original).all()

    # Assert labels
    names = [
        f"Dog[{d}]:Side[{s}]|day[{g}]" for g in [2, 4, 6] for d in [1, 2, 3] for s in ["L", "R"]
    ]
    assert dm.group.terms["Dog:Side|day"].labels == names


def test_prop_response(beetle):
    response = design_matrices("prop(y, n) ~ x", beetle).response
    assert response.kind == "proportion"
    assert response.design_vector.shape == (8, 2)
    assert (np.less_equal(response.design_vector[:, 0], response.design_vector[:, 1])).all()

    # Admit integer values for 'n'
    response = design_matrices("prop(y, 62) ~ x", beetle).response
    assert response.kind == "proportion"
    assert response.design_vector.shape == (8, 2)
    assert (np.less_equal(response.design_vector[:, 0], response.design_vector[:, 1])).all()

    # Use aliases
    response = design_matrices("proportion(y, n) ~ x", beetle).response
    assert response.kind == "proportion"
    assert response.design_vector.shape == (8, 2)
    assert (np.less_equal(response.design_vector[:, 0], response.design_vector[:, 1])).all()

    # Use aliases
    response = design_matrices("p(y, n) ~ x", beetle).response
    assert response.kind == "proportion"
    assert response.design_vector.shape == (8, 2)
    assert (np.less_equal(response.design_vector[:, 0], response.design_vector[:, 1])).all()


def test_prop_response_fails():
    # x larger than n
    with pytest.raises(ValueError):
        design_matrices("prop(x, n) ~ 1", pd.DataFrame({"x": [2, 3], "n": [1, 2]}))

    # x and/or n not integer
    with pytest.raises(ValueError):
        design_matrices("prop(x, n) ~ 1", pd.DataFrame({"x": [2, 3.3], "n": [4, 4]}))

    with pytest.raises(ValueError):
        design_matrices("prop(x, n) ~ 1", pd.DataFrame({"x": [2, 3], "n": [4.3, 4]}))

    # x not a variable name
    with pytest.raises(ValueError):
        design_matrices("prop(10, n) ~ 1", pd.DataFrame({"x": [2, 3], "n": [1, 2]}))

    # trials must be integer, not float
    with pytest.raises(ValueError):
        design_matrices("prop(x, 3.4) ~ 1", pd.DataFrame({"x": [2, 3], "n": [1, 2]}))


def test_categoric_responses():
    data = pd.DataFrame(
        {
            "y1": np.random.choice(["A", "B", "C"], size=30),
            "y2": np.random.choice(["A", "B"], size=30),
            "y3": np.random.choice(["Hi there", "Bye bye", "What??"], size=30),
            "x": np.random.normal(size=30),
        }
    )

    # Multi-level response. Response is a design matrix of dummies that span the intercept.
    response = design_matrices("y1 ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels == ["A", "B", "C"]
    assert response.binary is False
    assert response.success is None

    # Multi-level response, explicitly converted to binary
    response = design_matrices("y1['A'] ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels is None  # when binary, levels is None by design
    assert response.binary is True
    assert response.success == "A"

    # Response has two levels but it is not flagged as binary because it was not converted to that
    # XTODO: Revisit if this logic is fine
    response = design_matrices("y2 ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels == ["A", "B"]
    assert response.binary is False
    assert response.success is None

    # Binary response with explicit level
    response = design_matrices("y2['B'] ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels is None
    assert response.binary is True
    assert response.success == "B"

    # Binary response with explicit level passed as identifier
    response = design_matrices("y2[B] ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels is None
    assert response.binary is True
    assert response.success == "B"

    # Binary response with explicit level with spaces
    response = design_matrices("y3['Bye bye'] ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels is None
    assert response.binary is True
    assert response.success == "Bye bye"

    # Users trying to use nested brackets (WHY?)
    with pytest.raises(ParseError, match=re.escape("Are you using nested brackets? Why?")):
        design_matrices("y3[A[B]] ~ x", data)

    # Users try to pass a number to use a number
    with pytest.raises(
        ParseError, match=re.escape("Subset notation only allows a string or an identifer")
    ):
        design_matrices("y3[1] ~ x", data)


def test_binary_function():
    size = 100
    data = pd.DataFrame(
        {
            "y": np.random.randint(0, 5, size=size),
            "x": np.random.randint(5, 10, size=size),
            "g": np.random.choice(["a", "b", "c"], size=size),
        }
    )

    # String value
    term = design_matrices("y ~ binary(g, 'c')", data).common["binary(g, c)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["g"] == "c"))

    # Numeric value
    term = design_matrices("y ~ binary(x, 7)", data).common["binary(x, 7)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["x"] == 7))

    # Variable name
    # string
    m = "b"
    term = design_matrices("y ~ binary(g, m)", data).common["binary(g, m)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["g"] == m))

    # numeric
    z = 8
    term = design_matrices("y ~ binary(x, z)", data).common["binary(x, z)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["x"] == z))

    # Pass nothing
    term = design_matrices("y ~ binary(x)", data).common["binary(x)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["x"] == 5))

    # Values not found in the variable
    with pytest.raises(ValueError):
        design_matrices("y ~ binary(g, 'Not found')", data)

    with pytest.raises(ValueError):
        design_matrices("y ~ binary(x, 999)", data)


def test_B_function():
    size = 100
    data = pd.DataFrame(
        {
            "y": np.random.randint(0, 5, size=size),
            "x": np.random.randint(5, 10, size=size),
            "g": np.random.choice(["a", "b", "c"], size=size),
        }
    )

    # String value
    term = design_matrices("y ~ B(g, 'c')", data).common["B(g, c)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["g"] == "c"))

    # Numeric value
    term = design_matrices("y ~ B(x, 7)", data).common["B(x, 7)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["x"] == 7))

    # Variable name
    # string
    m = "b"
    term = design_matrices("y ~ B(g, m)", data).common["B(g, m)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["g"] == m))

    # numeric
    z = 8
    term = design_matrices("y ~ B(x, z)", data).common["B(x, z)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["x"] == z))

    # Pass nothing
    term = design_matrices("y ~ B(x)", data).common["B(x)"].squeeze()
    assert np.array_equal(np.where(term == 1), np.where(data["x"] == 5))

    # Values not found in the variable
    with pytest.raises(ValueError):
        design_matrices("y ~ B(g, 'Not found')", data)

    with pytest.raises(ValueError):
        design_matrices("y ~ B(x, 999)", data)


def test_C_function():
    size = 100
    rng = np.random.default_rng(1234)
    data = pd.DataFrame(
        {
            "x": rng.integers(5, 10, size=size),
            "g": rng.choice(["a", "b", "c"], size=size),
        }
    )

    term = design_matrices("C(x)", data).common.terms["C(x)"]
    assert term.kind == "categoric"
    assert term.levels == ["6", "7", "8", "9"]
    assert term.data.shape == (100, 4)

    levels = [6, 8, 5, 7, 9]
    term = design_matrices("C(x, levels=levels)", data).common.terms["C(x, levels = levels)"]
    assert term.kind == "categoric"
    assert term.levels == [str(level) for level in levels[1:]]
    assert term.data.shape == (100, 4)

    levels = ["b", "c", "a"]
    term = design_matrices("C(g, levels=levels)", data).common.terms["C(g, levels = levels)"]
    assert term.kind == "categoric"
    assert term.levels == levels[1:]
    assert term.data.shape == (100, 2)

    # Treatment encoding, the default
    t1 = design_matrices("C(x, Treatment)", data).common.terms["C(x, Treatment)"]
    t2 = design_matrices("C(x, Treatment())", data).common.terms["C(x, Treatment())"]
    t3 = design_matrices("C(x, Treatment(5))", data).common.terms["C(x, Treatment(5))"]
    t4 = design_matrices("C(x)", data).common.terms["C(x)"]

    assert np.array_equal(t1.data, t2.data)
    assert np.array_equal(t1.data, t3.data)
    assert np.array_equal(t1.data, t4.data)

    t1 = design_matrices("C(g, Treatment)", data).common.terms["C(g, Treatment)"]
    t2 = design_matrices("C(g, Treatment())", data).common.terms["C(g, Treatment())"]
    t3 = design_matrices("C(g, Treatment('a'))", data).common.terms["C(g, Treatment(a))"]
    t4 = design_matrices("C(g)", data).common.terms["C(g)"]

    assert np.array_equal(t1.data, t2.data)
    assert np.array_equal(t1.data, t3.data)
    assert np.array_equal(t1.data, t4.data)

    # Sum encoding, not the default
    t1 = design_matrices("C(x, Sum)", data).common.terms["C(x, Sum)"]
    t2 = design_matrices("C(x, Sum())", data).common.terms["C(x, Sum())"]
    t3 = design_matrices("C(x, Sum(9))", data).common.terms["C(x, Sum(9))"]

    assert np.array_equal(t1.data, t2.data)
    assert np.array_equal(t1.data, t3.data)

    t1 = design_matrices("C(g, Sum)", data).common.terms["C(g, Sum)"]
    t2 = design_matrices("C(g, Sum())", data).common.terms["C(g, Sum())"]
    t3 = design_matrices("C(g, Sum('c'))", data).common.terms["C(g, Sum(c))"]

    assert np.array_equal(t1.data, t2.data)
    assert np.array_equal(t1.data, t3.data)


# NOTE: B(g, 'c') is then looked up ad B(g, c).
# From the second it is impossible to tell if c is a literal or variable


def test_C_aliases():
    size = 100
    rng = np.random.default_rng(1234)
    data = pd.DataFrame(
        {
            "x": rng.integers(5, 10, size=size),
            "g": rng.choice(["a", "b", "c"], size=size),
        }
    )

    t1 = design_matrices("S(x)", data).common.terms["S(x)"]
    t2 = design_matrices("C(x, Sum)", data).common.terms["C(x, Sum)"]

    assert np.array_equal(t1.data, t2.data)

    t1 = design_matrices("T(x)", data).common.terms["T(x)"]
    t2 = design_matrices("C(x, Treatment)", data).common.terms["C(x, Treatment)"]

    assert np.array_equal(t1.data, t2.data)


def test_S_function():
    size = 100
    rng = np.random.default_rng(1234)
    data = pd.DataFrame(
        {
            "x": rng.integers(5, 10, size=size),
            "g": rng.choice(["a", "b", "c"], size=size),
        }
    )

    term = design_matrices("0 + S(x)", data).common.terms["S(x)"]
    assert term.labels == ["S(x)[mean]"] + [f"S(x)[{l}]" for l in [5, 6, 7, 8]]

    term = design_matrices("S(x, 7)", data).common.terms["S(x, 7)"]
    assert term.labels == [f"S(x, 7)[{l}]" for l in [5, 6, 8, 9]]

    # It still drops last level, no matter we changed the order
    levels = [9, 8, 7, 6, 5]
    term = design_matrices("S(x, levels = levels)", data).common.terms["S(x, levels = levels)"]
    assert term.labels == [f"S(x, levels = levels)[{l}]" for l in levels[:-1]]

    term = design_matrices("S(x, 6, levels)", data).common.terms["S(x, 6, levels)"]
    assert term.labels == [f"S(x, 6, levels)[{l}]" for l in [9, 8, 7, 5]]


def test_T_function():
    size = 100
    rng = np.random.default_rng(1234)
    data = pd.DataFrame(
        {
            "x": rng.integers(5, 10, size=size),
            "g": rng.choice(["a", "b", "c"], size=size),
        }
    )
    term = design_matrices("0 + T(x)", data).common.terms["T(x)"]
    assert term.labels == [f"T(x)[{l}]" for l in [5, 6, 7, 8, 9]]

    term = design_matrices("T(x, 7)", data).common.terms["T(x, 7)"]
    assert term.labels == [f"T(x, 7)[{l}]" for l in [5, 6, 8, 9]]

    # It still drops first level, no matter we changed the order
    levels = [9, 8, 7, 6, 5]
    term = design_matrices("T(x, levels = levels)", data).common.terms["T(x, levels = levels)"]
    assert term.labels == [f"T(x, levels = levels)[{l}]" for l in levels[1:]]

    term = design_matrices("T(x, 6, levels)", data).common.terms["T(x, 6, levels)"]
    assert term.labels == [f"T(x, 6, levels)[{l}]" for l in [9, 8, 7, 5]]


def test_offset():
    size = 100
    rng = np.random.default_rng(1234)
    data = pd.DataFrame(
        {
            "y": rng.integers(0, 5, size=size),
            "x": rng.integers(5, 10, size=size),
            "g": np.random.choice(["a", "b", "c"], size=size),
        }
    )

    dm = design_matrices("y ~ offset(x)", data)
    term = dm.common.terms["offset(x)"]
    assert term.kind == "offset"
    assert term.labels == ["offset(x)"]
    assert (dm.common["offset(x)"].flatten() == data["x"].values).all()

    with pytest.raises(
        ValueError, match=re.escape("offset() can only be used with numeric variables")
    ):
        design_matrices("y ~ offset(g)", data)

    with pytest.raises(ValueError, match=re.escape("offset() cannot be used as a response term.")):
        design_matrices("offset(y) ~ x", data)


def test_predict_prop(beetle):
    # If trials is a variable, new dataset must have that variable
    dm = design_matrices("prop(y, n) ~ x", beetle)
    result = dm.response.evaluate_new_data(pd.DataFrame({"n": [10, 10, 30, 30]}))
    assert (result == np.array([10, 10, 30, 30])).all()

    # If trials is a constant value, return that same value
    dm = design_matrices("prop(y, 70) ~ x", beetle)
    result = dm.response.evaluate_new_data(pd.DataFrame({"n": [10, 10, 30, 30]}))
    assert (result == np.array([70, 70, 70, 70])).all()


def test_predict_offset(beetle):
    # If offset is a variable, new dataset must have that variable
    dm = design_matrices("y ~ x + offset(x)", beetle)
    result = dm.common.evaluate_new_data(pd.DataFrame({"x": [1, 2, 3]}))["offset(x)"]
    assert (result == np.array([1, 2, 3])[:, np.newaxis]).all()

    # If offset is a constant value, return that same value
    dm = design_matrices("y ~ x + offset(10)", beetle)
    result = dm.common.evaluate_new_data(pd.DataFrame({"x": [1, 2, 3]}))["offset(10)"]
    assert (result == np.array([10, 10, 10])[:, np.newaxis]).all()


def test_builtin_names(data):
    # Test the case where we pass a builtin in a function
    def f(x, add):
        if add:
            return x + 1
        else:
            return x

    design_matrices("f(x1, False)", data)
    design_matrices("f(x1, True)", data)

    # Test the case where the variable does not exists
    with pytest.raises(KeyError):
        design_matrices("f(xn, True)", data)


def test_unary_operators(data):
    design_matrices("+x1", data)

    with pytest.raises(ResolverError, match="Unary negation can only be applied to '0' or '1'"):
        design_matrices("-x1", data)


def test_special_funciton_calls(data):
    def f(x):
        return x

    # visit quoted name
    design_matrices("f(`x1`)", data)

    # visit grouping expression
    design_matrices("f((x1 + x2))", data)

    # comparison
    design_matrices("f(x3 > 1)", data)

    # proportion not in response
    with pytest.raises(ValueError, match="can only be used as a response term"):
        design_matrices("proportion(x3, 10)", data)


def test_design_matrices_multiple_assignment(data):
    response, common, group = design_matrices("y ~ x1 + (x1|x3)", data)

    assert isinstance(response, ResponseVector)
    assert isinstance(common, CommonEffectsMatrix)
    assert isinstance(group, GroupEffectsMatrix)


def test_design_matrices_repr_and_str(data):
    dm = design_matrices("y ~ x1 + (x1|x3)", data)

    text = (
        "DesignMatrices\n\n"
        "                  (rows, cols)\n"
        "Response:                (20,)\n"
        "Common:                (20, 2)\n"
        "Group-specific:        (20, 8)\n\n"
        "Use .reponse, .common, or .group to access the different members."
    )

    assert repr(dm) == text
    assert str(dm) == text


def test_attempt_to_evaluate_non_proportion_response(data):
    response, _, _ = design_matrices("y ~ x1", data)

    with pytest.raises(
        ValueError, match="Can't evaluate response term with kind different to 'proportion'"
    ):
        response.evaluate_new_data(data)


def test_response_as_data_frame(data):
    response, _, _ = design_matrices("y ~ x1", data)

    response_as_df = response.as_dataframe()
    assert response_as_df.columns.tolist() == ["y"]
    assert (response_as_df["y"] == data["y"]).all()


def test_response_as_array(data):
    response, _, _ = design_matrices("y ~ x1", data)
    assert (np.asarray(response) == data["y"]).all()


def test_response_repr_and_str(data):
    response, _, _ = design_matrices("y ~ x1", data)
    text = (
        "ResponseVector  \n"
        "  name: y\n"
        "  kind: numeric\n"
        "  length: 20\n\n"
        "To access the actual design vector do 'np.array(this_obj)'"
    )
    assert str(response) == text
    assert repr(response) == text

    response, _, _ = design_matrices("g ~ x1", data)
    text = (
        "ResponseVector  \n"
        "  name: g\n"
        "  kind: categoric\n"
        "  length: 20\n"
        "  binary: False\n"
        "  levels: ['A', 'B']\n\n"
        "To access the actual design vector do 'np.array(this_obj)'"
    )
    assert str(response) == text
    assert repr(response) == text

    response, _, _ = design_matrices("g ~ x1", data)
    text = (
        "ResponseVector  \n"
        "  name: g\n"
        "  kind: categoric\n"
        "  length: 20\n"
        "  binary: False\n"
        "  levels: ['A', 'B']\n\n"
        "To access the actual design vector do 'np.array(this_obj)'"
    )
    assert str(response) == text
    assert repr(response) == text

    response, _, _ = design_matrices("g[A] ~ x1", data)
    text = (
        "ResponseVector  \n"
        "  name: g\n"
        "  kind: categoric\n"
        "  length: 20\n"
        "  binary: True\n"
        "  success: A\n\n"
        "To access the actual design vector do 'np.array(this_obj)'"
    )
    str(response) == text


def test_common_as_data_frame(data):
    _, common, _ = design_matrices("g ~ x1 + x2", data)
    common_as_dataframe = common.as_dataframe()
    assert common_as_dataframe.columns.tolist() == ["Intercept", "x1", "x2"]
    assert (common_as_dataframe["Intercept"] == 1).all()
    assert (common_as_dataframe["x1"] == data["x1"]).all()
    assert (common_as_dataframe["x2"] == data["x2"]).all()


def test_common_get_bad_key(data):
    _, common, _ = design_matrices("g ~ x1 + x2", data)

    with pytest.raises(ValueError, match="is not a valid term name"):
        common["blabla"]


def test_common_as_array(data):
    _, common, _ = design_matrices("g ~ x1 + x2", data)
    assert np.asarray(common).shape == (20, 3)


def test_common_repr_and_str(data):
    _, common, _ = design_matrices("y ~ x1 + f", data)
    text = (
        "CommonEffectsMatrix with shape (20, 3)\n"
        "Terms:  \n"
        "  Intercept  \n"
        "    kind: intercept\n"
        "    column: 0\n"
        "  x1  \n"
        "    kind: numeric\n"
        "    column: 1\n"
        "  f  \n"
        "    kind: categoric\n"
        "    levels: ['B']\n"
        "    column: 2\n\n"
        "To access the actual design matrix do 'np.array(this_obj)'"
    )
    assert str(common) == text
    assert repr(common) == text


def test_group_specific_get_bad_key(data):
    _, _, group = design_matrices("y ~ 1 + (x1|g)", data)

    with pytest.raises(ValueError, match="is not a valid term name"):
        group["blabla"]


def test_group_specific_as_array(data):
    _, _, group = design_matrices("y ~ 1 + (x1|g)", data)
    assert np.asarray(group).shape == (20, 4)


def test_group_specific_repr_and_str(data):
    _, _, group = design_matrices("y ~ 1 + (x1|g) + (h|g)", data)
    text = (
        "GroupEffectsMatrix with shape (20, 6)\n"
        "Terms:  \n"
        "  1|g  \n"
        "    kind: intercept\n"
        "    groups: ['A', 'B']\n"
        "    columns: 0:2\n"
        "  x1|g  \n"
        "    kind: numeric\n"
        "    groups: ['A', 'B']\n"
        "    columns: 2:4\n"
        "  h|g  \n"
        "    kind: categoric\n"
        "    groups: ['A', 'B']\n"
        "    levels: ['B']\n"
        "    columns: 4:6\n\n"
        "To access the actual design matrix do 'np.array(this_obj)'"
    )
    assert str(group) == text
    assert repr(group) == text


def test_formula_not_string(data):
    with pytest.raises(ValueError, match="'formula' must be a string."):
        design_matrices(True, data)


def test_data_not_data_frame(data):
    with pytest.raises(ValueError, match="'data' must be a pandas.DataFrame."):
        design_matrices("y ~ x", np.asarray(data))


def test_data_is_empty(data):
    with pytest.raises(ValueError, match="'data' does not contain any observation."):
        design_matrices("y ~ x", data.iloc[:0])


def test_wrong_na_action(data):
    with pytest.raises(ValueError, match="'na_action' must be either 'drop', 'error' or 'pass'"):
        design_matrices("y ~ x1", data, na_action="whatever")


def test_drop_na_rows(data):
    data["x3"] = data["x3"].replace({3: np.nan})
    dm = design_matrices("y ~ x3", data, na_action="drop")
    assert dm.common.as_dataframe().shape[0] == 15


def test_pass_na_rows(data):
    data["x3"] = data["x3"].replace({3: np.nan})
    dm = design_matrices("y ~ x3", data, na_action="pass")
    assert dm.common.as_dataframe().shape[0] == 20


def test_raise_na_rows(data):
    data["x3"] = data["x3"].replace({3: np.nan})
    with pytest.raises(ValueError, match="incomplete rows"):
        design_matrices("y ~ x3", data, na_action="error")


def test_design_matrices_categoric_call(data):
    def f(x):
        return pd.Categorical(x)

    def f_ordered(x):
        return pd.Categorical(x, categories=np.unique(x), ordered=True)

    # does not span intercept
    design_matrices("y ~ f(x3) + f_ordered(g)", data)

    # spans intercept
    dm = design_matrices("y ~ 0 + f(g)", data)

    # Evaluate new data
    dm.common.evaluate_new_data(data)


def test_extra_namespace(data):
    extra_namespace = {"myfunc": np.log}
    dm = design_matrices(
        "y ~ myfunc(x3) + x3", data, na_action="drop", extra_namespace=extra_namespace
    )
    df = dm.common.as_dataframe()
    assert df["myfunc(x3)"].equals(np.log(df["x3"]))
