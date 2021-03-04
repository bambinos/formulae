from numpy.core.numeric import allclose
import pytest

import numpy as np
import pandas as pd

from formulae.design_matrices import design_matrices

# TODO: See interaction names.. they don't always work as expected
@pytest.fixture(scope="module")
def data():
    np.random.seed(1234)
    size = 20
    data = pd.DataFrame(
        {
            "y": np.random.uniform(size=size),
            "x1": np.random.uniform(size=size),
            "x2": np.random.uniform(size=size),
            "x3": [1, 2, 3, 4] * 5,
            "f": np.random.choice(["A", "B"], size=size),
            "g": np.random.choice(["A", "B"], size=size),
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
    assert len(dm.common.terms_info) == 1
    assert dm.common.terms_info["Intercept"]["type"] == "Intercept"
    assert dm.common.terms_info["Intercept"]["full_names"] == ["Intercept"]
    assert all(dm.common.design_matrix == 1)
    assert dm.group == None


def test_group_specific_intercept_only(data):
    dm = design_matrices("y ~ 0 + (1|g)", data)
    assert len(dm.group.terms_info) == 1
    assert dm.group.terms_info["1|g"]["type"] == "Intercept"
    assert dm.group.terms_info["1|g"]["groups"] == ["A", "B"]
    assert dm.group.terms_info["1|g"]["full_names"] == ["1|g[A]", "1|g[B]"]
    assert dm.common == None


def test_common_predictor(data):
    dm = design_matrices("y ~ x1", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "x1"]
    assert dm.common.terms_info["x1"]["type"] == "numeric"
    assert dm.common.terms_info["x1"]["full_names"] == ["x1"]

    # uses alphabetic order
    # reference is the first value by default
    # reduced because we included intercept
    dm = design_matrices("y ~ f", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == sorted(list(data["f"].unique()))
    assert dm.common.terms_info["f"]["reference"] == sorted(list(data["f"].unique()))[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [
        f"f[{l}]" for l in sorted(data["f"].unique())[1:]
    ]


def test_categoric_encoding(data):
    # No intercept, one categoric predictor
    dm = design_matrices("y ~ 0 + f", data)
    assert list(dm.common.terms_info.keys()) == ["f"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == sorted(list(data["f"].unique()))
    assert dm.common.terms_info["f"]["reference"] == sorted(list(data["f"].unique()))[0]
    assert dm.common.terms_info["f"]["encoding"] == "full"
    assert dm.common.terms_info["f"]["full_names"] == [
        f"f[{l}]" for l in sorted(data["f"].unique())
    ]
    assert dm.common.design_matrix.shape == (20, 2)

    # Intercept, one categoric predictor
    dm = design_matrices("y ~ 1 + f", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == sorted(list(data["f"].unique()))
    assert dm.common.terms_info["f"]["reference"] == sorted(list(data["f"].unique()))[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [
        f"f[{l}]" for l in sorted(data["f"].unique())[1:]
    ]
    assert dm.common.design_matrix.shape == (20, 2)

    # No intercept, two additive categoric predictors
    dm = design_matrices("y ~ 0 + f + g", data)
    assert list(dm.common.terms_info.keys()) == ["f", "g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == sorted(list(data["f"].unique()))
    assert dm.common.terms_info["g"]["levels"] == sorted(list(data["g"].unique()))
    assert dm.common.terms_info["f"]["reference"] == sorted(list(data["f"].unique()))[0]
    assert dm.common.terms_info["g"]["reference"] == sorted(list(data["g"].unique()))[0]
    assert dm.common.terms_info["f"]["encoding"] == "full"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [
        f"f[{l}]" for l in sorted(data["f"].unique())
    ]
    assert dm.common.terms_info["g"]["full_names"] == [
        f"g[{l}]" for l in sorted(data["g"].unique())[1:]
    ]
    assert dm.common.design_matrix.shape == (20, 3)

    # Intercept, two additive categoric predictors
    dm = design_matrices("y ~ 1 + f + g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f", "g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == sorted(list(data["f"].unique()))
    assert dm.common.terms_info["g"]["levels"] == sorted(list(data["g"].unique()))
    assert dm.common.terms_info["f"]["reference"] == sorted(list(data["f"].unique()))[0]
    assert dm.common.terms_info["g"]["reference"] == sorted(list(data["g"].unique()))[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [
        f"f[{l}]" for l in sorted(data["f"].unique())[1:]
    ]
    assert dm.common.terms_info["g"]["full_names"] == [
        f"g[{l}]" for l in sorted(data["g"].unique())[1:]
    ]
    assert dm.common.design_matrix.shape == (20, 3)

    # No intercept, two categoric predictors with interaction
    dm = design_matrices("y ~ 0 + f + g + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["f", "g", "f:g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f"]["levels"] == sorted(list(data["f"].unique()))
    assert dm.common.terms_info["g"]["levels"] == sorted(list(data["g"].unique()))
    assert dm.common.terms_info["f"]["reference"] == sorted(list(data["f"].unique()))[0]
    assert dm.common.terms_info["g"]["reference"] == sorted(list(data["g"].unique()))[0]
    assert dm.common.terms_info["f"]["encoding"] == "full"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [
        f"f[{l}]" for l in sorted(data["f"].unique())
    ]
    assert dm.common.terms_info["g"]["full_names"] == [
        f"g[{l}]" for l in sorted(data["g"].unique())[1:]
    ]
    assert dm.common.terms_info["f:g"]["full_names"] == ["f[B]:g[B]"]
    assert dm.common.design_matrix.shape == (20, 4)

    # Intercept, two categoric predictors with interaction
    dm = design_matrices("y ~ 1 + f + g + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f", "g", "f:g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f"]["levels"] == sorted(list(data["f"].unique()))
    assert dm.common.terms_info["g"]["levels"] == sorted(list(data["g"].unique()))
    assert dm.common.terms_info["f"]["reference"] == sorted(list(data["f"].unique()))[0]
    assert dm.common.terms_info["g"]["reference"] == sorted(list(data["g"].unique()))[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [
        f"f[{l}]" for l in sorted(data["f"].unique())[1:]
    ]
    assert dm.common.terms_info["g"]["full_names"] == [
        f"g[{l}]" for l in sorted(data["g"].unique())[1:]
    ]
    assert dm.common.terms_info["f:g"]["full_names"] == ["f[B]:g[B]"]
    assert dm.common.design_matrix.shape == (20, 4)

    # No intercept, interaction between two categorics
    dm = design_matrices("y ~ 0 + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["f:g"]
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["full_names"] == [
        "f[A]:g[A]",
        "f[A]:g[B]",
        "f[B]:g[A]",
        "f[B]:g[B]",
    ]
    assert dm.common.design_matrix.shape == (20, 4)

    # Intercept, interaction between two categorics
    # It adds "g" -> It uses Patsy algorithm... look there if you're curious.
    dm = design_matrices("y ~ 1 + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "g", "f:g"]
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["full_names"] == ["f[B]:g[A]", "f[B]:g[B]"]
    assert dm.common.design_matrix.shape == (20, 4)

    # Same than before
    dm = design_matrices("y ~ 1 + g + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "g", "f:g"]
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["full_names"] == ["f[B]:g[A]", "f[B]:g[B]"]
    assert dm.common.design_matrix.shape == (20, 4)


def test_interactions(data):
    # These two models are the same
    dm = design_matrices("y ~ f * g", data)
    dm2 = design_matrices("y ~ f + g + f:g", data)
    assert compare_dicts(dm2.common.terms_info, dm.common.terms_info)

    # When no intercept too
    dm = design_matrices("y ~ 0 + f * g", data)
    dm2 = design_matrices("y ~ 0 + f + g + f:g", data)
    assert compare_dicts(dm2.common.terms_info, dm.common.terms_info)

    # Mix of numeric/categoric
    # "g" in "g" -> reduced
    # "g" in "x1:g" -> reduced because x1 is present in formula
    dm = design_matrices("y ~ x1 + g + x1:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "x1", "g", "x1:g"]
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["x1:g"]["terms"]["g"]["encoding"] == "reduced"

    # "g" in "g" -> reduced
    # "g" in "x1:g" -> full because x1 is not present in formula
    dm = design_matrices("y ~ g + x1:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "g", "x1:g"]
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["x1:g"]["terms"]["g"]["encoding"] == "full"

    # "g" in "x1:x2:g" is full, because x1:x2 is a new group and we don't have x1:x2 in the model
    dm = design_matrices("y ~ x1 + g + x1:g + x1:x2:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "x1", "g", "x1:g", "x1:x2:g"]
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["x1:g"]["terms"]["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["x1:x2:g"]["terms"]["g"]["encoding"] == "full"

    # "g" in "x1:x2:g" is reduced, because x1:x2 is a new group and we have x1:x2 in the model
    dm = design_matrices("y ~ x1 + g + x1:x2 + x1:g + x1:x2:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "x1", "g", "x1:x2", "x1:g", "x1:x2:g"]
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["x1:g"]["terms"]["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["x1:x2:g"]["terms"]["g"]["encoding"] == "reduced"

    # And now, since we don't have intercept, x1 and x1:x2 all "g" are full
    dm = design_matrices("y ~ 0 + g + x1:g + x1:x2:g", data)
    assert list(dm.common.terms_info.keys()) == ["g", "x1:g", "x1:x2:g"]
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["encoding"] == "full"
    assert dm.common.terms_info["x1:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["x1:x2:g"]["terms"]["g"]["encoding"] == "full"

    # Two numerics
    dm = design_matrices("y ~ x1:x2", data)
    assert "x1:x2" in dm.common.terms_info.keys()
    assert np.allclose(dm.common["x1:x2"][:, 0], data["x1"] * data["x2"])


def test_built_in_transformations(data):
    # {...} gets translated to I(...)
    dm = design_matrices("y ~ {x1 + x2}", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "I(x1 + x2)"]
    assert dm.common.terms_info["I(x1 + x2)"]["type"] == "call"
    assert np.allclose(
        dm.common["I(x1 + x2)"], np.atleast_2d((data["x1"] + data["x2"]).to_numpy()).T
    )

    dm2 = design_matrices("y ~ I(x1 + x2)", data)
    assert compare_dicts(dm.common.terms_info, dm2.common.terms_info)

    # center()
    dm = design_matrices("y ~ center(x1)", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "center(x1)"]
    assert dm.common.terms_info["center(x1)"]["type"] == "call"
    assert np.allclose(dm.common["center(x1)"].mean(), 0)

    # scale()
    dm = design_matrices("y ~ scale(x1)", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "scale(x1)"]
    assert dm.common.terms_info["scale(x1)"]["type"] == "call"
    assert np.allclose(dm.common["scale(x1)"].mean(), 0)
    assert np.allclose(dm.common["scale(x1)"].std(), 1)

    # standardize(), alias of scale()
    dm = design_matrices("y ~ standardize(x1)", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "standardize(x1)"]
    assert dm.common.terms_info["standardize(x1)"]["type"] == "call"
    assert np.allclose(dm.common["standardize(x1)"].mean(), 0)
    assert np.allclose(dm.common["standardize(x1)"].std(), 1)

    # C()
    # Intercept, no extra arguments, reference is first value observed
    dm = design_matrices("y ~ C(x3)", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "C(x3)"]
    assert dm.common.terms_info["C(x3)"]["type"] == "categoric"
    assert dm.common.terms_info["C(x3)"]["encoding"] == "reduced"
    assert dm.common.terms_info["C(x3)"]["reference"] == 1
    assert dm.common.terms_info["C(x3)"]["levels"] == [1, 2, 3, 4]
    assert dm.common.terms_info["C(x3)"]["full_names"] == ["C(x3)[2]", "C(x3)[3]", "C(x3)[4]"]

    # No intercept, no extra arguments
    dm = design_matrices("y ~ 0 + C(x3)", data)
    assert list(dm.common.terms_info.keys()) == ["C(x3)"]
    assert dm.common.terms_info["C(x3)"]["type"] == "categoric"
    assert dm.common.terms_info["C(x3)"]["encoding"] == "full"
    assert dm.common.terms_info["C(x3)"]["reference"] == 1
    assert dm.common.terms_info["C(x3)"]["levels"] == [1, 2, 3, 4]
    assert dm.common.terms_info["C(x3)"]["full_names"] == [
        "C(x3)[1]",
        "C(x3)[2]",
        "C(x3)[3]",
        "C(x3)[4]",
    ]

    # Specify reference -> it is converted into 0-1 variable
    dm = design_matrices("y ~ C(x3, 3)", data)
    assert dm.common.terms_info["C(x3, 3)"]["type"] == "call"
    assert dm.common.terms_info["C(x3, 3)"]["full_names"] == ["C(x3, 3)"]
    assert all(dm.common["C(x3, 3)"][:, 0] == np.where(data["x3"] == 3, 1, 0))

    # Specify levels, different to observed
    lvls = [3, 2, 4, 1]
    dm = design_matrices("y ~ C(x3, levels=lvls)", data)
    assert dm.common.terms_info["C(x3, levels = lvls)"]["type"] == "categoric"
    assert dm.common.terms_info["C(x3, levels = lvls)"]["reference"] == 3
    assert dm.common.terms_info["C(x3, levels = lvls)"]["levels"] == lvls

    # Pass a reference not in the data
    with pytest.raises(ValueError):
        dm = design_matrices("y ~ C(x3, 5)", data)

    # Pass categoric, remains unchanged
    dm = design_matrices("y ~ C(f)", data)
    dm2 = design_matrices("y ~ f", data)
    d1 = dm.common.terms_info["C(f)"]
    d2 = dm2.common.terms_info["f"]
    assert d1["type"] == d2["type"]
    assert d1["levels"] == d2["levels"]
    assert d1["reference"] == d2["reference"]
    assert d1["encoding"] == d2["encoding"]
    assert not d1["full_names"] == d2["full_names"]  # because one is 'C(f)' and other is 'f'
    assert all(dm.common["C(f)"] == dm2.common["f"])


def test_external_transformations(data):
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
    assert list(dm.common.terms_info.keys()) == [
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
