import sys

import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from formulae.transforms import TRANSFORMS, Proportion, Offset
from formulae.terms.call_utils import CallVarsExtractor


class Call:
    """Representation of a call in a model Term.

    This class and ``Variable`` are the atomic components of a model term.

    This object supports stateful transformations defined in ``formulae.transforms``.
    A transformation of this type defines its parameters the first time it is called,
    and then can be used to recompute the transformation with memorized parameter values.
    This behavior is useful when implementing a predict method and using transformations such
    as ``center(x)`` or ``scale(x)``. ``center(x)`` memorizes the value of the mean, and
    ``scale(x)`` memorizes both the mean and the standard deviation.

    Parameters
    ----------
    call: formulae.terms.call_resolver.LazyCall
        The call expression returned by the parser.
    is_response: bool
        Indicates whether this call represents a response. Defaults to ``False``.
    """

    def __init__(self, call, is_response=False):
        self.data = None
        self.eval_env = None
        self._intermediate_data = None
        self._type = None
        self.is_response = is_response
        self.call = call
        self.name = str(self.call)

    def __hash__(self):
        return hash(self.call)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.call == other.call

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def accept(self, visitor):
        """Accept method called by a visitor.

        Visitors are those available in call_utils.py, and are used to work with call terms.
        """
        return visitor.visitCallTerm(self)

    @property
    def var_names(self):
        """Returns the names of the variables involved in the call, not including the callee.

        This is used to determine which variables of the data set being used are actually used in
        the model. This allows us to subset the original data set and only raise errors regarding
        missing values when the missingness happens in variables used in the model.

        Uses a visitor of class ``CallVarsExtractor`` that walks through the components of the call
        and returns a list with the name of the variables in the call.

        Returns
        ----------
        result: list
            A list of strings with the names of the names of the variables in the call, not
            including the name of the callee.
        """
        return set(CallVarsExtractor(self).get())

    def set_type(self, data_mask, eval_env):
        """Evaluates function and determines the type of the result of the call.

        Evaluates the function call and sets the ``._type`` property to ``"numeric"`` or
        ``"categoric"`` depending on the type of the result. It also stores the intermediate result
        of the evaluation in ``._intermediate_data`` to prevent us from computing the same thing
        more than once.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from
        eval_env: EvalEnvironment
            The environment where values and functions are taken from.
        """

        self.eval_env = eval_env.with_outer_namespace(TRANSFORMS)
        x = self.call.eval(data_mask, self.eval_env)

        if is_numeric_dtype(x):
            self._type = "numeric"
        elif is_string_dtype(x) or is_categorical_dtype(x):
            self._type = "categoric"
        elif isinstance(x, Proportion):
            self._type = "proportion"
        elif isinstance(x, Offset):
            self._type = "offset"
            x.set_size(len(data_mask.index))
        else:
            raise ValueError(f"Call result is of an unrecognized type ({type(x)}).")
        self._intermediate_data = x

    def set_data(self, encoding=False):
        """Finishes the evaluation of the call according to its type.

        Evaluates the call according to its type and stores the result in ``.data``. It does not
        support multi-level categoric responses yet. If ``self.is_response`` is ``True`` and the
        variable is of a categoric type, this method returns a 1d array of 0-1 instead of a matrix.

        In practice, it just completes the evaluation that started with ``self.set_type()``.

        Parameters
        ----------
        encoding: bool
            Indicates if it uses full or reduced encoding when the type of the call is
            categoric. Omitted when the result of the call is numeric.
        """
        try:
            if self._type is None:
                raise ValueError("Call result type is not set.")
            if self._type not in ["numeric", "categoric", "proportion", "offset"]:
                raise ValueError(f"Call result is of an unrecognized type ({self._type}).")
            if self._type == "numeric":
                self.data = self._eval_numeric(self._intermediate_data)
            elif self._type == "categoric":
                self.data = self._eval_categoric(self._intermediate_data, encoding)
            elif self._type == "proportion":
                self.data = self._eval_proportion(self._intermediate_data)
            elif self._type == "offset":
                self.data = self._eval_offset(self._intermediate_data)
        except:
            print("Unexpected error while trying to evaluate a Call:", sys.exc_info()[0])
            raise

    def _eval_numeric(self, x):
        """Finishes evaluation of a numeric call.

        Converts the intermediate values of the call into a numpy array of shape ``(n, 1)``,
        where ``n`` is the number of observations. This method is used both in ``self.set_data``
        and in ``self.eval_new_data``.

        Parameters
        ----------
        x: np.ndarray or pd.Series
            The intermediate values resulting from the call.

        Returns
        ----------
        result: dict
            A dictionary with keys ``"value"`` and ``"type"``. The first contains the result of the
            evaluation, and the latter is equal to ``"numeric"``.
        """
        if isinstance(x, np.ndarray):
            value = x.flatten()
            if value.ndim > 1:
                raise ValueError(f"The result of {self.name} is not 1-dimensional.")
            value = value[:, np.newaxis]
        elif isinstance(x, pd.Series):
            value = x.to_numpy()[:, np.newaxis]
        else:
            raise ValueError(f"Call result is of an unrecognized type ({type(x)}).")
        return {"value": value, "type": "numeric"}

    def _eval_categoric(self, x, encoding):
        """Finishes evaluation of categoric call.

        First, it checks whether the intermediate evaluation returned is ordered. If not, it
        creates a category where the levels are the observed in the variable. They are sorted
        according to ``sorted()`` rules.

        Then, it determines the reference level as well as all the other levels. If the variable
        is a response, the value returned is a dummy with 1s for the reference level and 0s
        elsewhere. If it is not a response variable, it determines the matrix of dummies according
        to the levels and the encoding passed.

        Parameters
        ----------
        x: np.ndarray or pd.Series
            The intermediate values of the variable.
        encoding: bool
            Indicates if it uses full or reduced encoding.

        Returns
        ----------
        result: dict
            A dictionary with keys ``"value"``, ``"type"``, ``"levels"``, ``"reference"``, and
            ``"encoding"``. They represent the result of the evaluation, the type, which is
            ``"categoric"``, the levels observed in the variable, the level used as reference when
            using reduced encoding, and whether the encoding is ``"full"`` or ``"reduced"``.
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

    def _eval_proportion(self, proportion):
        if not self.is_response:
            raise ValueError("'prop()' can only be used in the context of a response term.")
        return {"value": proportion.eval(), "type": "proportion"}

    def _eval_offset(self, offset):
        if self.is_response:
            raise ValueError("'offset() cannot be used in the context of a response term.")
        return {"value": offset.eval(), "type": "offset"}

    def eval_new_data(self, data_mask):  # pylint: disable = inconsistent-return-statements
        """Evaluates the function call with new data.

        This method evaluates the function call within a new data mask. If the transformation
        applied is a stateful transformation, it uses the proper object that remembers all
        parameters or settings that may have been set in a first pass.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from

        Returns
        ----------
        result: np.array
            The rules for the shape of this array are the rules for ``self._eval_numeric()`` and
            ``self._eval_categoric()``. The first applies for numeric calls, the second for
            categoric ones.
        """
        if self._type in ["numeric", "categoric"]:
            x = self.call.eval(data_mask, self.eval_env)
            if self._type == "numeric":
                return self._eval_numeric(x)["value"]
            else:
                return self._eval_new_data_categoric(x)
        elif self._type == "proportion":
            if self._intermediate_data.trials_type == "constant":
                # Return value passed in the second component
                return np.ones((len(data_mask.index), 1)) * self.call.args[1].value
            else:
                # Extract name of the second component
                name = self.call.args[1].name
                values = data_mask[name]
                if isinstance(values, pd.Series):
                    values = values.values[:, np.newaxis]
                return values
        elif self._type == "offset":
            if self._intermediate_data.type == "constant":
                # Return value passed as the argument
                return np.ones((len(data_mask.index), 1)) * self.call.args[0].value
            else:
                # Extract name of the argument
                name = self.call.args[0].name
                values = data_mask[name]
                if isinstance(values, pd.Series):
                    values = values.values[:, np.newaxis]
                return values

    def _eval_new_data_categoric(self, x):
        """Evaluates the call with new data when the result of the call is categoric.

        This method also checks the levels observed in the new data frame are included within the
        set of the levels of the result of the original call If not, an error is raised.

        x: np.ndarray or pd.Series
            The intermediate values of the variable.

        Returns
        ----------
        result: np.array
            Numeric numpy array ``(n, p)``, where ``n`` is the number of observations and ``p`` the
            number of dummy variables used in the numeric representation of the categorical
            variable.
        """

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
