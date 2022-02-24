import sys

import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from formulae.categorical import Treatment


class Variable:
    """Representation of a variable in a model Term.

    This class and ``Call`` are the atomic components of a model term.

    Parameters
    ----------
    name: string
        The identifier of the variable.
    level: string
        The level to use as reference. Allows to use the notation ``variable["level"]`` to indicate
        which event should be model as success in binary response models. Can only be used with
        response terms. Defaults to ``None``.
    is_response: bool
        Indicates whether this variable represents a response. Defaults to ``False``.
    """

    def __init__(self, name, level=None, is_response=False):
        self.is_response = is_response
        self.name = name
        self.reference = level
        self.contrast_matrix = None
        self.kind = None
        self.levels = None
        self.spans_intercept = None
        self.value = None
        self._intermediate_data = None

    def __hash__(self):
        return hash((self.kind, self.name, self.reference))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.kind == other.kind
            and self.name == other.name
            and self.reference == other.reference
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        args = self.name
        if self.reference is not None:
            args += f", reference='{self.reference}'"
        return f"{self.__class__.__name__}({args})"

    @property
    def var_names(self):
        """Returns the name of the variable as a set.

        This is used to determine which variables of the data set being used are actually used in
        the model. This allows us to subset the original data set and only raise errors regarding
        missing values when the missingness happens in variables used in the model.
        """
        return {self.name}

    def set_type(self, data_mask):
        """Detemines the type of the variable.

        Looks for the name of the variable in ``data_mask`` and sets the ``.kind`` property to
        ``"numeric"`` or ``"categoric"`` depending on the type of the variable.
        It also stores the result of the intermediate evaluation in ``self._intermediate_data``.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from
        """
        x = data_mask[self.name]
        if is_numeric_dtype(x):
            self.kind = "numeric"
        elif is_string_dtype(x) or is_categorical_dtype(x):
            self.kind = "categoric"
        else:
            raise ValueError(f"Variable is of an unrecognized type ({type(x)}).")
        self._intermediate_data = x

    def set_data(self, spans_intercept=None):
        """Obtains and stores the final data object related to this variable.

        Parameters
        ----------
        spans_intercept: bool
            Indicates if the encoding of categorical variables spans the intercept or not.
            Omitted when the variable is numeric.
        """

        try:
            if self.kind is None:
                raise ValueError("Variable type is not set.")
            if self.kind not in ["numeric", "categoric"]:
                raise ValueError(f"Variable is of an unrecognized type ({self.kind}).")
            if self.kind == "numeric":
                self.eval_numeric(self._intermediate_data)
            elif self.kind == "categoric":
                self.eval_categoric(self._intermediate_data, spans_intercept)
        except:
            print("Unexpected error while trying to evaluate a Variable.", sys.exc_info()[0])
            raise

    def eval_numeric(self, x):
        """Finishes evaluation of a numeric variable.

        Converts the intermediate values in ``x`` into a 1d numpy array.

        Parameters
        ----------
        x: np.ndarray or pd.Series
            The intermediate values of the variable.
        """
        if isinstance(x, np.ndarray):
            self.value = x
        elif isinstance(x, pd.Series):
            self.value = x.values
        else:
            raise ValueError(f"Variable is of an unrecognized type ({type(x)}).")

    def eval_categoric(self, x, spans_intercept):
        """Finishes evaluation of a categoric variable.

        Converts the intermediate values in ``x`` into a numpy array of shape ``(n, p)``, where
        ``n`` is the number of observations and ``p`` the number of dummy variables used in the
        numeric representation of the categorical variable.

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

        # Result of 'variable[level]' is always binary
        if self.is_response and self.reference is not None:
            value = np.where(x == self.reference, 1, 0)
        else:
            # Treatment encoding by default
            treatment = Treatment()
            if spans_intercept:
                self.contrast_matrix = treatment.code_with_intercept(self.levels)
            else:
                self.contrast_matrix = treatment.code_without_intercept(self.levels)
            value = self.contrast_matrix.matrix[x.codes]

        self.value = value
        self.spans_intercept = spans_intercept

    def eval_new_data(self, data_mask):
        """Evaluates the variable with new data.

        This method evaluates the variable within a new data mask. If this object is categorical,
        original encoding is remembered (and checked) when carrying out the new evaluation.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from

        Returns
        ----------
        result: np.array
            The rules for the shape of this array are the rules for ``self.eval_numeric()`` and
            ``self.eval_categoric()``. The first applies for numeric variables, the second for
            categoric ones.
        """
        x = data_mask[self.name]
        if self.kind == "numeric":
            return self.eval_new_data_numeric(x)
        else:
            return self.eval_new_data_categoric(x)

    def eval_new_data_numeric(self, x):
        return np.asarray(x)

    def eval_new_data_categoric(self, x):
        """Evaluates the variable with new data when variable is categoric.

        This method also checks the levels observed in the new data frame are included within the
        set of the levels of the original data set. If not, an error is raised.

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

    @property
    def labels(self):
        """Obtain labels of the columns in the design matrix associated with this Variable"""
        labels = None
        if self.kind == "numeric":
            if self.value.ndim == 2 and self.value.shape[1] > 1:
                labels = [f"{self.name}[{i}]" for i in range(self.value.shape[1])]
            else:
                labels = [self.name]
        elif self.kind == "categoric":
            labels = [f"{self.name}[{label}]" for label in self.contrast_matrix.labels]

        return labels
