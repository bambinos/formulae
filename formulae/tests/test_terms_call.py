import pytest

import numpy as np
import pandas as pd

from formulae.eval import EvalEnvironment
from formulae.parser import Parser
from formulae.scanner import Scanner
from formulae.terms import Call
from formulae.terms.call_resolver import CallResolver


def call(expr):
    """Replicates the series of steps that occur when parsing a function call."""
    return Call(CallResolver(Parser(Scanner(expr).scan(False)).parse()).resolve())


def test_call_str():

    assert str(call("f(x)")) == "Call(f(x))"
    assert str(call("module.function(x, y)")) == "Call(module.function(x, y))"

    assert str(call("f(x)")) == repr(call("f(x)"))
    assert str(call("module.function(x, y)")) == repr(call("module.function(x, y)"))


def test_call_unrecognized_type():
    f = lambda x: x
    eval_env = EvalEnvironment.capture()
    with pytest.raises(ValueError):
        call("f(x)").set_type({"x": 1}, eval_env)

    with pytest.raises(ValueError):
        call("x").set_type({"x": [1, 2]}, eval_env)

    with pytest.raises(ValueError):
        call("x").set_type({"x": set([5, 6])}, eval_env)


def test_call_set_data_errors():
    f = lambda x: x
    x = call("f(x)")

    with pytest.raises(ValueError):
        x.set_data(True)

    with pytest.raises(ValueError):
        x._type = "hello"
        x.set_data(True)

    with pytest.raises(Exception):
        x._type = "categoric"
        x.set_data(True)


def test_call_eval_numeric():
    f = lambda x: x
    x = call("f(x)")

    arr = np.array([[1, 2, 3, 4]])
    series = pd.Series([1, 2, 3, 4])

    # Row vectors are transposed to column vectors
    assert np.array_equal(x._eval_numeric(arr)["value"], arr.T)
    assert np.array_equal(x._eval_numeric(series)["value"], arr.T)

    with pytest.raises(ValueError):
        x._eval_numeric([1, 2, 3])
