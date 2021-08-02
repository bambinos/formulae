import sys

import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype


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
        self.data = None
        self._intermediate_data = None
        self._type = None
        self.is_response = is_response
        self.name = name
        self.level = level

    def __hash__(self):
        return hash((self._type, self.name, self.level))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._type == other._type and self.name == other.name and self.level == other.level

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, level='{self.level}')"

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

        Looks for the name of the variable in ``data_mask`` and sets the ``.type_`` property to
        ``"numeric"`` or ``"categoric"`` depending on the type of the variable.
        It also stores the result of the intermediate evaluation in ``self._intermediate_data`` to
        save computing time later.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from
        """
        x = data_mask[self.name]
        if is_numeric_dtype(x):
            self._type = "numeric"
            if self.level is not None:
                raise ValueError("Subset notation can't be used with a numeric variable.")
        elif is_string_dtype(x) or is_categorical_dtype(x):
            self._type = "categoric"
        else:
            raise ValueError(f"Variable is of an unrecognized type ({type(x)}).")
        self._intermediate_data = x

    def set_data(self, encoding=None):
        """Obtains and stores the final data object related to this variable.

        The result is stored in ``self.data``.

        Parameters
        ----------
        encoding: bool
            Indicates if it uses full or reduced encoding when the type of the variable is
            categoric. Omitted when the variable is numeric.
        """

        try:
            if self._type is None:
                raise ValueError("Variable type is not set.")
            if self._type not in ["numeric", "categoric"]:
                raise ValueError(f"Variable is of an unrecognized type ({self._type}).")
            if self._type == "numeric":
                self.data = self._eval_numeric(self._intermediate_data)
            elif self._type == "categoric":
                self.data = self._eval_categoric(self._intermediate_data, encoding)
        except:
            print("Unexpected error while trying to evaluate a Variable.", sys.exc_info()[0])
            raise

    def _eval_numeric(self, x):
        """Finishes evaluation of a numeric variable.

        Converts the intermediate values in ``x`` into a numpy array of shape ``(n, 1)``,
        where ``n`` is the number of observations. This method is used both in ``self.set_data``
        and in ``self.eval_new_data``.

        Parameters
        ----------
        x: np.ndarray or pd.Series
            The intermediate values of the variable.

        Returns
        ----------
        result: dict
            A dictionary with keys ``"value"`` and ``"type"``. The first contains the result of the
            evaluation, and the latter is equal to ``"numeric"``.
        """
        if isinstance(x, np.ndarray):
            value = np.atleast_2d(x)
            if x.shape[0] == 1 and x.shape[1] > 1:
                value = value.T
        elif isinstance(x, pd.Series):
            value = np.atleast_2d(x.to_numpy()).T
        else:
            raise ValueError(f"Variable is of an unrecognized type ({type(x)}).")
        return {"value": value, "type": "numeric"}

    def _eval_categoric(self, x, encoding):
        """Finishes evaluation of a categoric variable.

        Converts the intermediate values in ``x`` into a numpy array of shape ``(n, p)``, where
        ``n`` is the number of observations and ``p`` the number of dummy variables used in the
        numeric representation of the categorical variable.

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
        # If not ordered, we make it ordered.
        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            categories = sorted(x.unique().tolist())
            cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if self.is_response:
            # Will be binary, no matter how many levels
            if self.level is not None:
                reference = self.level
                value = np.where(x == reference, 1, 0)[:, np.newaxis]
            # Is binary, model first event
            elif len(x.unique()) == 2:
                value = np.where(x == reference, 1, 0)[:, np.newaxis]
            # Isn't binary, no level has been passed, return codes.
            else:
                value = pd.Categorical(x).codes[:, np.newaxis]
        else:
            # Not always we receive a bool, so we need to check.
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
            The rules for the shape of this array are the rules for ``self._eval_numeric()`` and
            ``self._eval_categoric()``. The first applies for numeric variables, the second for
            categoric ones.

        """
        if self.data is None:
            raise ValueError("self.data is None. This error shouldn't have happened!")
        x = data_mask[self.name]
        if self._type == "numeric":
            return self._eval_numeric(x)["value"]
        else:
            return self._eval_new_data_categoric(x)

    def _eval_new_data_categoric(self, x):
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
