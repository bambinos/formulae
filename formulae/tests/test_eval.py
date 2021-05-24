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
