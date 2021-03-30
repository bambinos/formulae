import pytest

import numpy as np
import pandas as pd

from formulae.eval import EvalEnvironment
from formulae.parser import Parser
from formulae.scanner import Scanner
from formulae.terms import Variable, Call, Term, Model


def test_new_data_numeric():
    data = pd.DataFrame({"x": [10, 10, 10]})
    var_expr = Parser(Scanner("x").scan(False)).parse()
    var_term = Variable(var_expr.name.lexeme, var_expr.level)
    var_term.set_type(data)
    var_term.set_data()
    assert (var_term.data["value"].T == [10, 10, 10]).all()
    data = pd.DataFrame({"x": [1, 2, 3]})
    assert (var_term.eval_new_data(data).T == [1, 2, 3]).all()


def test_new_data_numeric_stateful_transform():
    # The center() transformation remembers the value of the mean
    # of the first dataset passed, which is 10.
    eval_env = EvalEnvironment.capture(0)
    data = pd.DataFrame({"x": [10, 10, 10]})
    call_term = Call(Parser(Scanner("center(x)").scan(False)).parse())
    call_term.set_type(data, eval_env)
    call_term.set_data()
    assert (call_term.data["value"].T == [0, 0, 0]).all()
    data = pd.DataFrame({"x": [1, 2, 3]})
    assert (call_term.eval_new_data(data, eval_env).T == [-9.0, -8.0, -7.0]).all()


def test_new_data_categoric():
    data = pd.DataFrame({"x": ["A", "B", "C"]})

    # Full rank encoding
    var_expr = Parser(Scanner("x").scan(False)).parse()
    var_term = Variable(var_expr.name.lexeme, var_expr.level)
    var_term.set_type(data)
    var_term.set_data(encoding=True)
    assert (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == var_term.data["value"]).all()

    data = pd.DataFrame({"x": ["B", "C"]})
    assert (var_term.eval_new_data(data) == np.array([[0, 1, 0], [0, 0, 1]])).all()

    # It remembers it saw "A", "B", and "C", but not "D".
    # So when you pass a new level, it raises a ValueError.
    with pytest.raises(ValueError):
        data = pd.DataFrame({"x": ["B", "C", "D"]})
        var_term.eval_new_data(data)

    # The same with reduced encoding
    data = pd.DataFrame({"x": ["A", "B", "C"]})
    var_expr = Parser(Scanner("x").scan(False)).parse()
    var_term = Variable(var_expr.name.lexeme, var_expr.level)
    var_term.set_type(data)
    var_term.set_data()
    assert (np.array([[0, 0], [1, 0], [0, 1]]) == var_term.data["value"]).all()

    data = pd.DataFrame({"x": ["A", "C"]})
    assert (var_term.eval_new_data(data) == np.array([[0, 0], [0, 1]])).all()

    # It remembers it saw "A", "B", and "C", but not "D".
    # So when you pass a new level, it raises a ValueError.
    with pytest.raises(ValueError):
        data = pd.DataFrame({"x": ["B", "C", "D"]})
        var_term.eval_new_data(data)


def test_new_data_categoric_stateful_transform():
    eval_env = EvalEnvironment.capture(0)
    data = pd.DataFrame({"x": [1, 2, 3]})

    # Full rank encoding
    call_term = Call(Parser(Scanner("C(x)").scan(False)).parse())
    call_term.set_type(data, eval_env)
    call_term.set_data(encoding=True)
    assert (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == call_term.data["value"]).all()

    data = pd.DataFrame({"x": [2, 3]})
    assert (call_term.eval_new_data(data, eval_env) == np.array([[0, 1, 0], [0, 0, 1]])).all()

    with pytest.raises(ValueError):
        data = pd.DataFrame({"x": [2, 3, 4]})
        call_term.eval_new_data(data, eval_env)

    # The same with reduced encoding
    data = pd.DataFrame({"x": [1, 2, 3]})
    call_term = Call(Parser(Scanner("C(x)").scan(False)).parse())
    call_term.set_type(data, eval_env)
    call_term.set_data()
    assert (np.array([[0, 0], [1, 0], [0, 1]]) == call_term.data["value"]).all()

    data = pd.DataFrame({"x": [1, 3]})
    assert (call_term.eval_new_data(data, eval_env) == np.array([[0, 0], [0, 1]])).all()

    # It remembers it saw "A", "B", and "C", but not "D".
    # So when you pass a new level, it raises a ValueError.
    with pytest.raises(ValueError):
        data = pd.DataFrame({"x": [2, 3, 4]})
        call_term.eval_new_data(data, eval_env)
