import sys

import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from formulae.categorical import ENCODINGS, CategoricalBox, Treatment
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
        self.call = call
        self.is_response = is_response
        self.name = str(self.call)
        self.contrast_matrix = None
        self.env = None
        self.kind = None
        self.levels = None
        self.spans_intercept = None
        self.value = None
        self._intermediate_data = None

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

    def set_type(self, data_mask, env):
        """Evaluates function and determines the type of the result of the call.

        Evaluates the function call and sets the ``.kind`` property to ``"numeric"`` or
        ``"categoric"`` depending on the type of the result. It also stores the intermediate result
        of the evaluation in ``._intermediate_data`` to prevent us from computing the same thing
        more than once.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.
        """

        self.env = env.with_outer_namespace({**TRANSFORMS, **ENCODINGS})
        x = self.call.eval(data_mask, self.env)

        if is_numeric_dtype(x):
            self.kind = "numeric"
        elif is_string_dtype(x) or is_categorical_dtype(x) or isinstance(x, CategoricalBox):
            self.kind = "categoric"
        elif isinstance(x, Offset):
            self.kind = "offset"
            x.set_size(len(data_mask.index))
        elif isinstance(x, Proportion):
            self.kind = "proportion"
        else:
            raise ValueError(f"Call result is of an unrecognized type ({type(x)}).")
        self._intermediate_data = x

    def set_data(self, spans_intercept=False):
        """Finishes the evaluation of the call according to its type.

        It does not support multi-level categoric responses yet.
        If ``self.is_response`` is ``True`` and the variable is of a categoric type, this method
        returns a 1d array of 0-1 instead of a matrix.
        # XTODO: Fix previous point
        In practice, it just completes the evaluation that started with ``self.set_type()``.

        Parameters
        ----------
        spans_intercept: bool
            Indicates if the encoding of categorical variables spans the intercept or not.
            Omitted when the variable is numeric.
        """
        try:
            if self.kind is None:
                raise ValueError("Call result type is not set.")
            if self.kind == "numeric":
                self.eval_numeric(self._intermediate_data)
            elif self.kind == "categoric":
                if isinstance(self._intermediate_data, CategoricalBox):
                    self.eval_categorical_box(self._intermediate_data, spans_intercept)
                else:
                    self.eval_categoric(self._intermediate_data, spans_intercept)
            elif self.kind == "offset":
                self.eval_offset(self._intermediate_data)
            elif self.kind == "proportion":
                self.eval_proportion(self._intermediate_data)
            else:
                raise ValueError(f"Call result is of an unrecognized type ({self.kind}).")
        except:
            print("Unexpected error while trying to evaluate a Call:", sys.exc_info()[0])
            raise

    def eval_numeric(self, x):
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
            A dictionary with keys ``"value"`` and ``"kind"``. The first contains the result of the
            evaluation, and the latter is equal to ``"numeric"``.
        """
        if isinstance(x, np.ndarray):
            self.value = x
        elif isinstance(x, pd.Series):
            self.value = x.values
        else:
            raise ValueError(f"Call result is of an unrecognized type ({type(x)}).")

    def eval_categoric(self, x, spans_intercept):
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
        spans_intercept: bool
            Indicates if the encoding of categorical variables spans the intercept or not.
            Omitted when the variable is numeric.
        """

        # If not ordered, we make it ordered.
        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            categories = sorted(np.unique(x).tolist())
            dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
            x = pd.Categorical(x).astype(dtype)
        else:
            x = pd.Categorical(x)

        self.levels = x.categories.tolist()

        treatment = Treatment()
        if spans_intercept:
            self.contrast_matrix = treatment.code_with_intercept(self.levels)
        else:
            self.contrast_matrix = treatment.code_without_intercept(self.levels)

        self.value = self.contrast_matrix.matrix[x.codes]
        self.spans_intercept = spans_intercept

    def eval_categorical_box(self, box, spans_intercept):
        data = box.data
        levels = box.levels
        contrast = box.contrast

        if contrast is None:
            contrast = Treatment()

        if levels is None:
            categories = sorted(list(set(data)))
        else:
            categories = levels

        dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
        data = pd.Categorical(data).astype(dtype)
        self.levels = categories

        if spans_intercept:
            self.contrast_matrix = contrast.code_with_intercept(categories)
        else:
            self.contrast_matrix = contrast.code_without_intercept(categories)

        self.value = self.contrast_matrix.matrix[data.codes]
        self.spans_intercept = spans_intercept

    def eval_proportion(self, proportion):
        if not self.is_response:
            raise ValueError("'proportion()' can only be used as a response term.")
        self.value = proportion.eval()

    def eval_offset(self, offset):
        if self.is_response:
            raise ValueError("offset() cannot be used as a response term.")
        self.value = offset.eval()

    def eval_new_data(self, data_mask):
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
            The rules for the shape of this array are the rules for ``self.eval_numeric()`` and
            ``self.eval_categoric()``. The first applies for numeric calls, the second for
            categoric ones.
        """
        if self.kind in ["numeric", "categoric"]:
            x = self.call.eval(data_mask, self.env)
            if self.kind == "numeric":
                result = self.eval_new_data_numeric(x)
            elif isinstance(x, CategoricalBox):
                result = self.eval_new_data_categorical_box(x)
            else:
                result = self.eval_new_data_categoric(x)
        elif self.kind == "offset":
            result = self.eval_new_data_offset(data_mask)
        elif self.kind == "proportion":
            result = self.eval_new_data_proportion(data_mask)

        return result

    def eval_new_data_numeric(self, x):
        return np.asarray(x)

    def eval_new_data_categoric(self, x):
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
        new_data_levels = set(x)
        original_levels = set(self.levels)
        difference = new_data_levels - original_levels

        if not difference:
            idxs = pd.Categorical(x, categories=self.levels).codes
            return self.contrast_matrix.matrix[idxs]
        else:
            difference = [str(x) for x in difference]
            raise ValueError(
                f"The levels {', '.join(difference)} in '{self.name}' are not present in "
                "the original data set."
            )

    def eval_new_data_categorical_box(self, x):
        return self.eval_new_data_categoric(x.data)

    def eval_new_data_offset(self, data_mask):
        if self._intermediate_data.kind == "constant":
            # Return value passed as the argument
            result = np.ones(len(data_mask.index)) * self.call.args[0].value
        else:
            # Extract name of the argument
            name = self.call.args[0].name
            values = data_mask[name]
            if isinstance(values, pd.Series):
                values = values.values
            result = values
        return result

    def eval_new_data_proportion(self, data_mask):
        if self._intermediate_data.trials_type == "constant":
            # Return value passed in the second component
            result = np.ones(len(data_mask.index)) * self.call.args[1].value
        else:
            # Extract name of the second component
            name = self.call.args[1].name
            values = data_mask[name]
            if isinstance(values, pd.Series):
                values = values.values
            result = values
        return result

    @property
    def labels(self):
        """Obtain labels of the columns in the design matrix associated with this Call"""
        labels = None
        if self.kind in ["numeric", "offset"]:
            if self.value.ndim == 2 and self.value.shape[1] > 1:
                labels = [f"{self.name}[{i}]" for i in range(self.value.shape[1])]
            else:
                labels = [self.name]
        elif self.kind == "categoric":
            labels = [f"{self.name}[{label}]" for label in self.contrast_matrix.labels]

        return labels
