# VarLookupDict is obtained from Patsy.
# EvalEnvironment inspired from EvalEnvironment in Patsy.
# TODO: Include Patsy license

import inspect
import pandas as pd

class VarLookupDict(object):
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

    def find_modules(self):
        l = [key for keys in self.keys() for key in keys if inspect.ismodule(self[key])]
        return list(set(l))

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self._dicts)

class EvalEnvironment:
    def __init__(self, n=0):
        self.n = self._check_n(n)
        self._namespaces = self.capture()

    def capture(self):
        frame = inspect.currentframe()
        try:
            for i in range(self.n + 2):
                frame = frame.f_back
                if frame is None:
                    raise ValueError("call-stack is not that deep!")
            # Important to have locals before globals!!
            # See VarLookupDict.__getitem__()
            return [frame.f_locals, frame.f_globals]
        finally:
            del frame

    @property
    def namespace(self):
        """A dict-like object that can be used to look up variables accessible
        from the encapsulated environment."""
        return VarLookupDict(self._namespaces)

    def _check_n(self, n):
        msg = "n must be a positive integer"
        if not isinstance(n, int):
            raise ValueError(msg)
        if n < 0:
            raise ValueError(msg)
        return n

def eval_in_data_mask(expr, data=None, n=1):
    data_dict = {}
    names_conflict = False

    env = EvalEnvironment(n=n)

    if data is not None:
        if isinstance(data, pd.DataFrame):
           data_dict_inner = data.reset_index(drop=True).to_dict('series')
           data_dict = {'__DATA__': data_dict_inner}
           env_modules = env.namespace.find_modules()
           names_conflict = any(key in env_modules for key in data_dict.keys())
        else:
            raise ValueError("data must be a pandas DataFrame")

    if names_conflict:
        raise ValueError("At least one column has a name conflicting with an imported module")

    return eval(expr, {}, VarLookupDict([data_dict] + env._namespaces))

