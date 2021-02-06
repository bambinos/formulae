from copy import deepcopy


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def flatten_list(nested_list):
    """Flatten a nested list"""
    nested_list = deepcopy(nested_list)
    while nested_list:
        sublist = nested_list.pop(0)
        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist
