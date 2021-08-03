import pytest

import numpy as np
import pandas as pd

from formulae.matrices import design_matrices

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
    assert dm.common.terms_info["Intercept"]["type"] == "intercept"
    assert dm.common.terms_info["Intercept"]["full_names"] == ["Intercept"]
    assert all(dm.common.design_matrix == 1)
    assert dm.group == None


def test_group_specific_intercept_only(data):
    dm = design_matrices("y ~ 0 + (1|g)", data)
    assert len(dm.group.terms_info) == 1
    assert dm.group.terms_info["1|g"]["type"] == "intercept"
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


def test_categoric_encoding_with_numeric_interaction():
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
            "h": np.random.choice(["A", "B"], size=size),
            "j": np.random.choice(["A", "B"], size=size),
        }
    )
    dm = design_matrices("y ~ x1 + x2 + f:g + h:j:x2", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "x1", "x2", "g", "f:g", "j", "h:j:x2"]
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["full_names"] == ["f[B]:g[A]", "f[B]:g[B]"]
    assert dm.common.terms_info["j"]["encoding"] == "reduced"
    assert dm.common.terms_info["h:j:x2"]["terms"]["h"]["encoding"] == "reduced"
    assert dm.common.terms_info["h:j:x2"]["terms"]["j"]["encoding"] == "full"
    assert dm.common.terms_info["h:j:x2"]["terms"]["x2"]["type"] == "numeric"


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


def test_built_in_transforms(data):
    # {...} gets translated to I(...)
    dm = design_matrices("y ~ {x1 + x2}", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "I(x1 + x2)"]
    assert dm.common.terms_info["I(x1 + x2)"]["type"] == "numeric"
    assert np.allclose(
        dm.common["I(x1 + x2)"], np.atleast_2d((data["x1"] + data["x2"]).to_numpy()).T
    )

    dm2 = design_matrices("y ~ I(x1 + x2)", data)
    assert compare_dicts(dm.common.terms_info, dm2.common.terms_info)

    # center()
    dm = design_matrices("y ~ center(x1)", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "center(x1)"]
    assert dm.common.terms_info["center(x1)"]["type"] == "numeric"
    assert np.allclose(dm.common["center(x1)"].mean(), 0)

    # scale()
    dm = design_matrices("y ~ scale(x1)", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "scale(x1)"]
    assert dm.common.terms_info["scale(x1)"]["type"] == "numeric"
    assert np.allclose(dm.common["scale(x1)"].mean(), 0)
    assert np.allclose(dm.common["scale(x1)"].std(), 1)

    # standardize(), alias of scale()
    dm = design_matrices("y ~ standardize(x1)", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "standardize(x1)"]
    assert dm.common.terms_info["standardize(x1)"]["type"] == "numeric"
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


def test_categoric_group_specific():
    data = pd.DataFrame(
        {
            "BP": np.random.normal(size=30),
            "BMI": np.random.normal(size=30),
            "age_grp": np.random.choice([0, 1, 2], size=30),
        }
    )
    dm = design_matrices("BP ~ 0 + (C(age_grp)|BMI)", data)
    list(dm.group.terms_info.keys()) == ["1|BMI", "C(age_grp)[1]|BMI", "C(age_grp)[2]|BMI"]

    dm = design_matrices("BP ~ 0 + (0 + C(age_grp)|BMI)", data)
    list(dm.group.terms_info.keys()) == [
        "C(age_grp)[0]|BMI",
        "C(age_grp)[1]|BMI",
        "C(age_grp)[2]|BMI",
    ]


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

    # Assert full names
    names = [f"day[{d}]|{g}" for g in [1, 2, 3] for d in [2, 4, 6]]
    assert dm.group.terms_info["day|Dog"]["full_names"] == names
    names = [f"1|Side[{s}]" for s in ["L", "R"]]
    assert dm.group.terms_info["1|Side"]["full_names"] == names
    names = [f"1|Side:Dog[{s}:{d}]" for s in ["L", "R"] for d in [1, 2, 3]]
    assert dm.group.terms_info["1|Side:Dog"]["full_names"] == names

    # Another design matrix
    dm = design_matrices("(0 + Dog:Side | day)", pixel)
    dog_and_side_by_day = dm.group["Dog:Side|day"]

    # Assert values in the design matrix
    assert (dog_and_side_by_day == dog_and_side_by_day_original).all()

    # Assert full names
    names = [f"Dog[{d}]:Side[{s}]|{g}" for g in [2, 4, 6] for d in [1, 2, 3] for s in ["L", "R"]]
    assert dm.group.terms_info["Dog:Side|day"]["full_names"] == names


def test_prop_response():
    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )

    response = design_matrices("prop(y, n) ~ x", data).response
    assert response.type == "proportion"
    assert response.design_vector.shape == (8, 2)
    assert (np.less_equal(response.design_vector[:, 0], response.design_vector[:, 1])).all()

    # Admit integer values for 'n'
    response = design_matrices("prop(y, 62) ~ x", data).response
    assert response.type == "proportion"
    assert response.design_vector.shape == (8, 2)
    assert (np.less_equal(response.design_vector[:, 0], response.design_vector[:, 1])).all()

    # Use aliases
    response = design_matrices("proportion(y, n) ~ x", data).response
    assert response.type == "proportion"
    assert response.design_vector.shape == (8, 2)
    assert (np.less_equal(response.design_vector[:, 0], response.design_vector[:, 1])).all()

    # Use aliases
    response = design_matrices("p(y, n) ~ x", data).response
    assert response.type == "proportion"
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
            "x": np.random.normal(size=30),
        }
    )

    # Multi-level response
    response = design_matrices("y1 ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1, 2]
    assert response.levels == ["A", "B", "C"]
    assert response.binary is False
    assert response.baseline == "A"
    assert response.success is None

    # Multi-level response, explicitly converted to binary
    response = design_matrices("y1['A'] ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels == ["A", "B", "C"]
    assert response.binary is True
    assert response.baseline is None
    assert response.success == "A"

    # Default binary response
    response = design_matrices("y2 ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels == ["A", "B"]
    assert response.binary is True
    assert response.baseline is None
    assert response.success == "A"

    # Binary response with explicit level
    response = design_matrices("y2['B'] ~ x", data).response
    assert list(np.unique(response.design_vector)) == [0, 1]
    assert response.levels == ["A", "B"]
    assert response.binary is True
    assert response.baseline is None
    assert response.success == "B"


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


def test_C_function():
    size = 100
    data = pd.DataFrame(
        {
            "y": np.random.randint(0, 5, size=size),
            "x": np.random.randint(5, 10, size=size),
            "g": np.random.choice(["a", "b", "c"], size=size),
        }
    )

    term = design_matrices("y ~ C(x)", data).common.terms_info["C(x)"]
    assert term["type"] == "categoric"
    assert term["levels"] == [5, 6, 7, 8, 9]
    assert term["reference"] == 5

    term = design_matrices("y ~ C(x, 7)", data).common.terms_info["C(x, 7)"]
    assert term["type"] == "categoric"
    assert term["levels"] == [7, 5, 6, 8, 9]
    assert term["reference"] == 7

    l = [6, 8, 5, 7, 9]
    term = design_matrices("y ~ C(x, levels=l)", data).common.terms_info["C(x, levels = l)"]
    assert term["type"] == "categoric"
    assert term["levels"] == l
    assert term["reference"] == 6

    term = design_matrices("y ~ C(g)", data).common.terms_info["C(g)"]
    assert term["type"] == "categoric"
    assert term["levels"] == ["a", "b", "c"]
    assert term["reference"] == "a"

    term = design_matrices("y ~ C(g, 'c')", data).common.terms_info["C(g, c)"]
    assert term["type"] == "categoric"
    assert term["levels"] == ["c", "a", "b"]
    assert term["reference"] == "c"

    l = ["b", "c", "a"]
    term = design_matrices("y ~ C(g, levels=l)", data).common.terms_info["C(g, levels = l)"]
    assert term["type"] == "categoric"
    assert term["levels"] == l
    assert term["reference"] == "b"

    with pytest.raises(ValueError):
        design_matrices("y ~ C(g, 'c', levels=l)", data)


def test_offset():
    size = 100
    data = pd.DataFrame(
        {
            "y": np.random.randint(0, 5, size=size),
            "x": np.random.randint(5, 10, size=size),
            "g": np.random.choice(["a", "b", "c"], size=size),
        }
    )

    dm = design_matrices("y ~ offset(x)", data)
    term = dm.common.terms_info["offset(x)"]
    assert term["type"] == "offset"
    assert term["full_names"] == ["offset(x)"]
    assert (dm.common["offset(x)"].flatten() == data["x"]).all()

    with pytest.raises(ValueError):
        design_matrices("y ~ offset(g)", data)

    with pytest.raises(ValueError):
        design_matrices("offset(y) ~ x", data)


def test_predict_prop():
    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )

    # If trials is a variable, new dataset must have that variable
    dm = design_matrices("prop(y, n) ~ x", data)
    result = dm.response._evaluate_new_data(pd.DataFrame({"n": [10, 10, 30, 30]}))
    assert (result == np.array([10, 10, 30, 30])[:, np.newaxis]).all()

    # If trials is a constant value, return that same value
    dm = design_matrices("prop(y, 70) ~ x", data)
    result = dm.response._evaluate_new_data(pd.DataFrame({"n": [10, 10, 30, 30]}))
    print(result)
    assert (result == np.array([70, 70, 70, 70])[:, np.newaxis]).all()


def test_predict_offset():
    data = pd.DataFrame(
        {
            "x": np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.8610, 1.8839]),
            "n": np.array([59, 60, 62, 56, 63, 59, 62, 60]),
            "y": np.array([6, 13, 18, 28, 52, 53, 61, 60]),
        }
    )

    # If offset is a variable, new dataset must have that variable
    dm = design_matrices("y ~ x + offset(x)", data)
    result = dm.common._evaluate_new_data(pd.DataFrame({"x": [1, 2, 3]}))["offset(x)"]
    assert (result == np.array([1, 2, 3])[:, np.newaxis]).all()

    # If offset is a constant value, return that same value
    dm = design_matrices("y ~ x + offset(10)", data)
    result = dm.common._evaluate_new_data(pd.DataFrame({"x": [1, 2, 3]}))["offset(10)"]
    assert (result == np.array([10, 10, 10])[:, np.newaxis]).all()
