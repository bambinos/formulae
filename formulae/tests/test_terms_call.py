import pytest

import numpy as np
import pandas as pd

from formulae.environment import Environment
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
    env = Environment.capture()
    with pytest.raises(ValueError):
        call("f(x)").set_type({"x": 1}, env)

    with pytest.raises(ValueError):
        call("x").set_type({"x": [1, 2]}, env)

    with pytest.raises(ValueError):
        call("x").set_type({"x": set([5, 6])}, env)


def test_call_set_data_errors():
    f = lambda x: x
    x = call("f(x)")

    with pytest.raises(ValueError):
        x.set_data(True)

    with pytest.raises(ValueError):
        x.kind = "hello"
        x.set_data(True)

    with pytest.raises(Exception):
        x.kind = "categoric"
        x.set_data(True)


def test_call_eval_numeric():
    f = lambda x: x
    x = call("f(x)")

    arr = np.array([1, 2, 3, 4])
    series = pd.Series([1, 2, 3, 4])

    x.eval_numeric(arr)
    assert np.array_equal(x.value, arr)

    x.eval_numeric(series)
    assert np.array_equal(x.value, arr)

    with pytest.raises(ValueError):
        x.eval_numeric([1, 2, 3])
