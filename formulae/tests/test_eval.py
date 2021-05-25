import pytest
from formulae.eval import VarLookupDict, EvalEnvironment


def test_varlookup_get():
    lookup = VarLookupDict([{"a": 1, "b": 2}, {"a": 2, "c": 3}])
    assert lookup["a"] == 1
    assert lookup["b"] == 2
    assert lookup["c"] == 3
    with pytest.raises(KeyError):
        lookup["d"]


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


def test_varlookup_get():
    lookup = VarLookupDict([{"a": 1, "b": 2}, {"a": 2, "c": 3}])
    assert lookup.get("a") == 1
    assert lookup.get("b") == 2
    assert lookup.get("c") == 3
    assert lookup.get("d") == None


def test_evalenv_namespace():
    a = 1
    b = "hello friend"
    c = [1, 2, 3]
    d = {"a": 1, "b": 2}

    eval_env = EvalEnvironment.capture()

    assert eval_env.namespace["a"] == a
    assert eval_env.namespace["b"] == b
    assert eval_env.namespace["c"] == c
    assert eval_env.namespace["d"] == d


def test_evalenv_with_outer_namespace():
    a = 1
    b = "hello friend"

    eval_env = EvalEnvironment.capture()

    assert eval_env.namespace["a"] == a
    assert eval_env.namespace["b"] == b
    assert "c" not in eval_env.namespace
    assert "d" not in eval_env.namespace

    eval_env = eval_env.with_outer_namespace({"c": [1, 2, 3], "d": {"a": 1, "b": 2}})

    assert eval_env.namespace["c"] == [1, 2, 3]
    assert eval_env.namespace["d"] == {"a": 1, "b": 2}


def test_evalenv_eval():
    a = 1
    b = 2
    eval_env = EvalEnvironment.capture()
    assert eval_env.eval("a + b") == 3
    assert eval_env.eval("a + b", {"b": 3}) == 4


def test_evalenv_capture():
    eval_env = EvalEnvironment.capture(0, 0)
    eval_env = EvalEnvironment.capture(1, 0)
    eval_env = EvalEnvironment.capture(0, 1)
    EvalEnvironment.capture(eval_env)
    with pytest.raises(TypeError):
        EvalEnvironment.capture("blah")
    with pytest.raises(ValueError):
        EvalEnvironment.capture(100)


def test_evalenv_subset():
    a = 1
    b = "hello friend"
    c = [1, 2, 3]
    d = {"a": 1, "b": 2}

    eval_env = EvalEnvironment.capture()

    assert "a" in eval_env.namespace
    assert "b" in eval_env.namespace
    assert "c" in eval_env.namespace
    assert "d" in eval_env.namespace

    eval_env = eval_env.subset(["a", "c"])
    assert eval_env.namespace["a"] == a
    assert eval_env.namespace["c"] == c
    assert "b" not in eval_env.namespace
    assert "d" not in eval_env.namespace


def test_evalenv_equality():
    a = 1
    b = "hello friend"
    c = [1, 2, 3]
    d = {"a": 1, "b": 2}
    # The comparison is made in terms of the ids of the namespaces
    eval_env = EvalEnvironment.capture()
    eval_env2 = EvalEnvironment.capture()
    assert eval_env == eval_env2

    eval_env_outer = EvalEnvironment.capture(1)
    assert eval_env != eval_env_outer
