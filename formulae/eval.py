# VarLookUpDict and EvalEnvironment are taken from Patsy library.
# For more info see: https://github.com/pydata/patsy/blob/master/patsy/eval.py
# pylint: disable=relative-beyond-top-level
import inspect
import numbers


class VarLookupDict:
    def __init__(self, dicts):
        self._dicts = [{}] + list(dicts)

    def __getitem__(self, key):
        for d in self._dicts:
            try:
                return d[key]
            except KeyError:
                pass
        raise KeyError(key)

    def __setitem__(self, key, value):
        self._dicts[0][key] = value

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return [list(d.keys()) for d in self._dicts]

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._dicts)


class EvalEnvironment:
    """Represents a Python execution environment.
    Encapsulates a namespace for variable lookup
    """

    def __init__(self, namespaces):
        self._namespaces = list(namespaces)

    @property
    def namespace(self):
        """A dict-like object that can be used to look up variables accessible
        from the encapsulated environment."""
        return VarLookupDict(self._namespaces)

    def with_outer_namespace(self, outer_namespace):
        return self.__class__(self._namespaces + [outer_namespace])

    def eval(self, expr, inner_namespace={}):  # pylint: disable = dangerous-default-value
        return eval(  # pylint: disable = eval-used
            expr,
            {},
            VarLookupDict([inner_namespace] + self._namespaces),
        )

    @classmethod
    def capture(cls, eval_env=0, reference=0):
        if isinstance(eval_env, cls):
            return eval_env
        elif isinstance(eval_env, numbers.Integral):
            depth = eval_env + reference
        else:
            raise TypeError(
                "Parameter 'eval_env' must be either an integer "
                "or an instance of EvalEnvironment."
            )
        frame = inspect.currentframe()
        try:
            for _ in range(depth + 1):
                if frame is None:
                    raise ValueError("call-stack is not that deep!")
                frame = frame.f_back
            return cls([frame.f_locals, frame.f_globals])
        finally:
            del frame

    def subset(self, names):
        """Creates a new, flat EvalEnvironment that contains only the variables specified."""
        vld = VarLookupDict(self._namespaces)
        new_ns = dict((name, vld[name]) for name in names)
        return EvalEnvironment([new_ns])

    def _namespace_ids(self):
        return [id(n) for n in self._namespaces]

    def __eq__(self, other):
        return (
            isinstance(other, EvalEnvironment) and self._namespace_ids() == other._namespace_ids()
        )

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((EvalEnvironment, tuple(self._namespace_ids())))
