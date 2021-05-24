import numpy as np
from formulae.utils import listify, flatten_list, get_interaction_matrix


def test_listify():
    assert listify(None) == []
    assert listify([]) == []
    assert listify(()) == ()
    assert listify([1, 2, 3]) == [1, 2, 3]
    assert listify((1, 2, 3)) == (1, 2, 3)
    assert isinstance(listify({"a": 1, "b": 2}), list)
    isinstance(listify(set((1, 2, 3))), list)


def test_flatten_list():
    l = list(flatten_list([[1, 2, 3], [1, 2, 3]]))
    assert l == [1, 2, 3, 1, 2, 3]
    l = list(flatten_list([["a", "b", ["c", ["d"]]]]))
    assert l == ["a", "b", "c", "d"]
    l = list(flatten_list([[], []]))
    assert l == []


def test_get_interaction_matrix():
    x = np.array([[1, 2, 3], [4, 5, 6]]).T
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    result = np.array([[1, 0, 0, 4, 0, 0], [0, 2, 0, 0, 5, 0], [0, 0, 3, 0, 0, 6]])
    assert (get_interaction_matrix(x, y) == result).all()
