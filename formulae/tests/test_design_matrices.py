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
            "f": np.random.choice(["A", "B"], size=size),
            "g": np.random.choice(["A", "B"], size=size),
        }
    )
    return data


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

    # uses order given in data
    # reference is the first value by default
    # reduced because we included intercept
    dm = design_matrices("y ~ f", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == list(data["f"].unique())
    assert dm.common.terms_info["f"]["reference"] == list(data["f"].unique())[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [f"f[{l}]" for l in data["f"].unique()[1:]]


def test_categoric_encoding(data):
    # No intercept, one categoric predictor
    dm = design_matrices("y ~ 0 + f", data)
    assert list(dm.common.terms_info.keys()) == ["f"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == list(data["f"].unique())
    assert dm.common.terms_info["f"]["reference"] == list(data["f"].unique())[0]
    assert dm.common.terms_info["f"]["encoding"] == "full"
    assert dm.common.terms_info["f"]["full_names"] == [f"f[{l}]" for l in data["f"].unique()]
    assert dm.common.design_matrix.shape == (20, 2)

    # Intercept, one categoric predictor
    dm = design_matrices("y ~ 1 + f", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == list(data["f"].unique())
    assert dm.common.terms_info["f"]["reference"] == list(data["f"].unique())[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [f"f[{l}]" for l in data["f"].unique()[1:]]
    assert dm.common.design_matrix.shape == (20, 2)

    # No intercept, two additive categoric predictors
    dm = design_matrices("y ~ 0 + f + g", data)
    assert list(dm.common.terms_info.keys()) == ["f", "g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == list(data["f"].unique())
    assert dm.common.terms_info["g"]["levels"] == list(data["g"].unique())
    assert dm.common.terms_info["f"]["reference"] == list(data["f"].unique())[0]
    assert dm.common.terms_info["g"]["reference"] == list(data["g"].unique())[0]
    assert dm.common.terms_info["f"]["encoding"] == "full"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [f"f[{l}]" for l in data["f"].unique()]
    assert dm.common.terms_info["g"]["full_names"] == [f"g[{l}]" for l in data["g"].unique()[1:]]
    assert dm.common.design_matrix.shape == (20, 3)

    # Intercept, two additive categoric predictors
    dm = design_matrices("y ~ 1 + f + g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f", "g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f"]["levels"] == list(data["f"].unique())
    assert dm.common.terms_info["g"]["levels"] == list(data["g"].unique())
    assert dm.common.terms_info["f"]["reference"] == list(data["f"].unique())[0]
    assert dm.common.terms_info["g"]["reference"] == list(data["g"].unique())[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [f"f[{l}]" for l in data["f"].unique()[1:]]
    assert dm.common.terms_info["g"]["full_names"] == [f"g[{l}]" for l in data["g"].unique()[1:]]
    assert dm.common.design_matrix.shape == (20, 3)

    # No intercept, two categoric predictors with interaction
    dm = design_matrices("y ~ 0 + f + g + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["f", "g", "f:g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f"]["levels"] == list(data["f"].unique())
    assert dm.common.terms_info["g"]["levels"] == list(data["g"].unique())
    assert dm.common.terms_info["f"]["reference"] == list(data["f"].unique())[0]
    assert dm.common.terms_info["g"]["reference"] == list(data["g"].unique())[0]
    assert dm.common.terms_info["f"]["encoding"] == "full"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [f"f[{l}]" for l in data["f"].unique()]
    assert dm.common.terms_info["g"]["full_names"] == [f"g[{l}]" for l in data["g"].unique()[1:]]
    assert dm.common.terms_info["f:g"]["full_names"] == ["f[A]:g[B]"]
    assert dm.common.design_matrix.shape == (20, 4)

    # Intercept, two categoric predictors with interaction
    dm = design_matrices("y ~ 1 + f + g + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "f", "g", "f:g"]
    assert dm.common.terms_info["f"]["type"] == "categoric"
    assert dm.common.terms_info["g"]["type"] == "categoric"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f"]["levels"] == list(data["f"].unique())
    assert dm.common.terms_info["g"]["levels"] == list(data["g"].unique())
    assert dm.common.terms_info["f"]["reference"] == list(data["f"].unique())[0]
    assert dm.common.terms_info["g"]["reference"] == list(data["g"].unique())[0]
    assert dm.common.terms_info["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f"]["full_names"] == [f"f[{l}]" for l in data["f"].unique()[1:]]
    assert dm.common.terms_info["g"]["full_names"] == [f"g[{l}]" for l in data["g"].unique()[1:]]
    assert dm.common.terms_info["f:g"]["full_names"] == ["f[A]:g[B]"]
    assert dm.common.design_matrix.shape == (20, 4)

    # No intercept, interaction between two categorics
    dm = design_matrices("y ~ 0 + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["f:g"]
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["full_names"] == ['f[B]:g[A]', 'f[B]:g[B]', 'f[A]:g[A]', 'f[A]:g[B]']
    assert dm.common.design_matrix.shape == (20, 4)

    # Intercept, interaction between two categorics
    # It adds "g" -> It uses Patsy algorithm... look there if you're curious.
    dm = design_matrices("y ~ 1 + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "g", "f:g"]
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["full_names"] == ['f[A]:g[A]', 'f[A]:g[B]']
    assert dm.common.design_matrix.shape == (20, 4)

    # Same than before
    dm = design_matrices("y ~ 1 + g + f:g", data)
    assert list(dm.common.terms_info.keys()) == ["Intercept", "g", "f:g"]
    assert dm.common.terms_info["g"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["type"] == "interaction"
    assert dm.common.terms_info["f:g"]["terms"]["f"]["encoding"] == "reduced"
    assert dm.common.terms_info["f:g"]["terms"]["g"]["encoding"] == "full"
    assert dm.common.terms_info["f:g"]["full_names"] == ['f[A]:g[A]', 'f[A]:g[B]']
    assert dm.common.design_matrix.shape == (20, 4)
