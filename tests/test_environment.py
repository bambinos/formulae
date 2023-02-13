import pytest
from formulae.environment import VarLookupDict, Environment


def test_varlookup_get():
    lookup = VarLookupDict([{"a": 1, "b": 2}, {"a": 2, "c": 3}])
    assert lookup["a"] == 1
    assert lookup["b"] == 2
    assert lookup["c"] == 3
    with pytest.raises(KeyError):
        lookup["d"]

    assert lookup.get("a") == 1
    assert lookup.get("b") == 2
    assert lookup.get("c") == 3
    assert lookup.get("d") == None


def test_varlookup_set():
    lookup = VarLookupDict([{"a": 1, "b": 2}, {"a": 2, "c": 3}])
    assert lookup["a"] == 1
    lookup["a"] = 2
    assert lookup["a"] == 2
    lookup["f"] = 1
    assert lookup["f"] == 1


def test_varlookup_contains():
    lookup = VarLookupDict([{"a": 1, "b": 2}, {"a": 2, "c": 3}])
    assert "a" in lookup
    assert "d" not in lookup


def testevalenv_namespace():
    a = 1
    b = "hello friend"
    c = [1, 2, 3]
    d = {"a": 1, "b": 2}

    env = Environment.capture()

    assert env.namespace["a"] == a
    assert env.namespace["b"] == b
    assert env.namespace["c"] == c
    assert env.namespace["d"] == d


def test_evalenv_with_outer_namespace():
    a = 1
    b = "hello friend"

    env = Environment.capture()

    assert env.namespace["a"] == a
    assert env.namespace["b"] == b
    assert "c" not in env.namespace
    assert "d" not in env.namespace

    env = env.with_outer_namespace({"c": [1, 2, 3], "d": {"a": 1, "b": 2}})

    assert env.namespace["c"] == [1, 2, 3]
    assert env.namespace["d"] == {"a": 1, "b": 2}


def test_evalenv_capture():
    env = Environment.capture(0, 0)
    env = Environment.capture(1, 0)
    env = Environment.capture(0, 1)
    Environment.capture(env)
    with pytest.raises(TypeError):
        Environment.capture("blah")
    with pytest.raises(ValueError):
        Environment.capture(100)


def test_evalenv_equality():
    a = 1
    b = "hello friend"
    c = [1, 2, 3]
    d = {"a": 1, "b": 2}
    # The comparison is made in terms of the ids of the namespaces
    env = Environment.capture()
    env2 = Environment.capture()
    assert env == env2

    env_outer = Environment.capture(1)
    assert env != env_outer
