import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from formulae.eval import eval_in_data_mask
from formulae.transforms import STATEFUL_TRANSFORMS
from formulae.terms.call_utils import (
    CallEvalPrinter,
    CallNamePrinter,
    CallVarsExtractor,
    get_data_mask_names,
)


class Call:
    """Atomic component of a Term that is a call.

    This object supports stateful transformations defined in formulae.transforms.
    A transformation of this type defines its parameters the first time it is called,
    and then can be used to recompute the transformation with memorized parameter values.
    This behavior is useful when implementing a predict method and using transformations such
    as ``center(x)`` or ``scale(x)``.

    Parameters
    ----------
    expr: formulae.expr.Call
        The call expression returned by the parser.
    """

    def __init__(self, expr, is_response=False):
        self.data = None
        self._intermediate_data = None
        self._type = None
        self.is_response = is_response
        self.callee = expr.callee.name.lexeme
        self.args = expr.args
        self.name = self._name_str()
        if self.callee in STATEFUL_TRANSFORMS.keys():
            self.stateful_transform = STATEFUL_TRANSFORMS[self.callee]()
        else:
            self.stateful_transform = None

    def __hash__(self):
        return hash((self.callee, self.name, self.stateful_transform))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.callee == other.callee and self.args == other.args

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def accept(self, visitor):
        """Accept method called by a visitor.

        Visitors are those available in call_utils.py, and are used to work with call terms.
        """
        return visitor.visitCallTerm(self)

    def _eval_str(self, names):
        """Generates the string used to evaluate the call."""
        return CallEvalPrinter(self, names).print()

    def _name_str(self):
        """Generates a string that reproduces the call and is used as name."""
        return CallNamePrinter(self).print()

    @property
    def var_names(self):
        """Returns the names of the variables involved in the call, not including the callee."""
        return set(CallVarsExtractor(self).get())

    def set_type(self, data_mask, eval_env):
        """Evaluates function and determines the type of the result of the call.

        Evaluates the function call and sets the ``._type`` property to ``"numeric"`` or
        ``"categoric"`` depending on the type of the result. It also stores the intermediate result
        of the evaluation in ``._intermediate_data`` to prevent us from computing the same thing
        more than once.
        """
        # Q: How to set non data dependent parameters?
        names = get_data_mask_names(data_mask)
        if self.stateful_transform is not None:
            # Adds the stateful transform to the environment this is evaluated
            eval_env = eval_env.with_outer_namespace({self.callee: self.stateful_transform})
        x = eval_in_data_mask(self._eval_str(names), data_mask, eval_env)
        if is_numeric_dtype(x):
            self._type = "numeric"
        elif is_string_dtype(x) or is_categorical_dtype(x) or isinstance(x, dict):
            self._type = "categoric"
        else:
            raise ValueError(f"Call result is of an unrecognized type ({type(x)}).")
        self._intermediate_data = x

    def set_data(self, encoding=False):
        """Completes evaluation of the call according to its type.

        Evaluates the call according to its type and stores the result in ``.data``. It does not
        support multi-level categoric responses yet. If ``self.is_response`` is ``True`` and the
        variable is of a categoric type, this method returns a 1d array of 0-1 instead of a matrix.

        In practice, it just completes the evaluation that started with ``self.set_type()``.
        """
        # Workaround: var names present in 'data' are taken from '__DATA__['col']
        # the rest are left as they are and looked up in the upper namespace
        if self._type is None:
            raise ValueError("Call result type is not set.")
        if self._type not in ["numeric", "categoric"]:
            raise ValueError(f"Call result is of an unrecognized type ({self._type}).")
        if self._type == "numeric":
            self.data = self._eval_numeric(self._intermediate_data)
        else:
            self.data = self._eval_categoric(self._intermediate_data, encoding)

    def _eval_numeric(self, x):
        """Finishes evaluation of a numeric call.

        This method ensures the returned type is a column vector (i.e. shape is (n, 1))
        """
        if isinstance(x, np.ndarray):
            value = np.atleast_2d(x)
            if x.shape[0] == 1 and len(x) > 1:
                value = value.T
        elif isinstance(x, pd.Series):
            value = np.atleast_2d(x.to_numpy()).T
        else:
            raise ValueError(f"Call result is of an unrecognized type ({type(x)}).")
        return {"value": value, "type": "numeric"}

    def _eval_categoric(self, x, encoding):
        # Note: Uncompleted docs.
        """Finishes evaluation of categoric call.

        Parameters
        ----------
         encoding: list or bool
            Determines whether to use reduced or full rank encoding.

        First, it checks whether the intermediate evaluation returned is ordered. If not, it
        creates a category where the levels are the observed in the variable. They are sorted
        according to ``sorted()`` rules.

        Then, it determines the reference level as well as all the other levels. If the variable
        is a response, the value returned is a dummy with 1s for the reference level and 0s
        elsewhere. If it is not a response variable, it determines the matrix of dummies according
        to the levels and the encoding passed.
        """

        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            categories = sorted(x.unique().tolist())
            cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if self.is_response:
            value = np.atleast_2d(np.where(x == reference, 1, 0)).T
            encoding = None
        else:
            if isinstance(encoding, list):
                encoding = encoding[0]
            if isinstance(encoding, dict):
                encoding = encoding[self.name]
            if encoding:
                value = pd.get_dummies(x).to_numpy()
                encoding = "full"
            else:
                value = pd.get_dummies(x, drop_first=True).to_numpy()
                encoding = "reduced"
        return {
            "value": value,
            "type": "categoric",
            "levels": levels,
            "reference": reference,
            "encoding": encoding,
        }

    def eval_new_data(self, data_mask, eval_env):
        """Evaluates the function call with new data.

        This method evaluates the function call within a new data mask. If the transformation
        applied is a stateful transformation, it uses the proper object that remembers all
        parameters or settings that may have been set in a first pass.
        """
        names = get_data_mask_names(data_mask)
        if self.stateful_transform is not None:
            eval_env = eval_env.with_outer_namespace({self.callee: self.stateful_transform})
        x = eval_in_data_mask(self._eval_str(names), data_mask, eval_env)
        if self._type == "numeric":
            return self._eval_numeric(x)["value"]
        else:
            return self._eval_new_data_categoric(x)

    def _eval_new_data_categoric(self, x):
        if self.is_response:
            return np.atleast_2d(np.where(x == self.data["reference"], 1, 0)).T
        else:
            # Raise error if passing a level that was not observed.
            new_data_levels = pd.Categorical(x).dtype.categories.tolist()
            if set(new_data_levels).issubset(set(self.data["levels"])):
                series = pd.Categorical(x, categories=self.data["levels"])
                drop_first = self.data["encoding"] == "reduced"
                return pd.get_dummies(series, drop_first=drop_first).to_numpy()
            else:
                raise ValueError(
                    f"At least one of the levels for '{self.name}' in the new data was "
                    "not present in the original data set."
                )
