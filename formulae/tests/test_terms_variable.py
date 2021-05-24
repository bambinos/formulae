import pytest

import numpy as np
import pandas as pd

from formulae.terms import Variable


def test_variable_str():

    assert str(Variable("a")) == "Variable(a, level='None')"
    assert str(Variable("a", "hi")) == "Variable(a, level='hi')"

    assert repr(Variable("a")) == str(Variable("a"))
    assert repr(Variable("a", "hi")) == str(Variable("a", "hi"))


def test_variable_subset_notation():
    with pytest.raises(ValueError):
        Variable("x", "hi").set_type({"x": np.array([1, 2])})


def test_variable_unrecognized_type():
    with pytest.raises(ValueError):
        Variable("x").set_type({"x": 1})

    with pytest.raises(ValueError):
        Variable("x").set_type({"x": [1, 2]})

    with pytest.raises(ValueError):
        Variable("x").set_type({"x": {"a": 1}})


def test_variable_set_data_errors():
    x = Variable("x")
    with pytest.raises(ValueError):
        x.set_data(True)

    with pytest.raises(ValueError):
        x._type = "hello"
        x.set_data(True)

    with pytest.raises(Exception):
        x._type = "categoric"
        x.set_data(True)


def test_variable_eval_numeric():
    x = Variable("x")
    arr = np.array([[1, 2, 3, 4]])
    series = pd.Series([1, 2, 3, 4])
    # Row vectors are transposed to column vectors
    assert np.array_equal(x._eval_numeric(arr)["value"], arr.T)

    assert np.array_equal(x._eval_numeric(series)["value"], arr.T)

    with pytest.raises(ValueError):
        x._eval_numeric([1, 2, 3])
