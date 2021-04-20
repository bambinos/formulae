import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype


class Variable:
    """Atomic component of a Term"""

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

        This ends up being used in design_matrices.py to determine which variables of the data
        passed are present in the model and subset and raise proper errors.
        """
        return {self.name}

    def set_type(self, data_mask):
        """Detemines the type of the variable.

        Looks for the name of the variable in ``data`` and sets the ``.type_`` property to
        ``"numeric"`` or ``"categoric"`` depending on the type of the variable.
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
        """
        if self._type is None:
            raise ValueError("Variable type is not set.")
        if self._type not in ["numeric", "categoric"]:
            raise ValueError(f"Variable is of an unrecognized type ({self._type}).")
        if self._type == "numeric":
            self.data = self._eval_numeric(self._intermediate_data)
        elif self._type == "categoric":
            self.data = self._eval_categoric(self._intermediate_data, encoding)
        else:
            raise ValueError("Unexpected error while trying to evaluate a Variable.")

    def _eval_numeric(self, x):
        if isinstance(x, np.ndarray):
            value = np.atleast_2d(x)
            if x.shape[0] == 1 and len(x) > 1:
                value = value.T
        elif isinstance(x, pd.Series):
            value = np.atleast_2d(x.to_numpy()).T
        else:
            raise ValueError(f"Variable is of an unrecognized type ({type(x)}).")
        return {"value": value, "type": "numeric"}

    def _eval_categoric(self, x, encoding):
        """
        It does not support multi-level categoric responses yet. If ``self.is_response`` is ``True``
        and the variable is of a categoric type, the value element of the dictionary returned is a
        1d array of 0-1 instead of a matrix.
        """
        # If not ordered, we make it ordered.
        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            categories = sorted(x.unique().tolist())
            cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if self.is_response:
            if self.level is not None:
                reference = self.level
            value = np.atleast_2d(np.where(x == reference, 1, 0)).T
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

    def eval_new_data(self, data_mask):
        """Evaluates the variable with new data.

        This method evaluates the variable within a new data mask. If this object is categorical,
        original encoding is remembered when carrying out the new evaluation
        """
        if self.data is None:
            raise ValueError("self.data is None. This error shouldn't have happened!")
        x = data_mask[self.name]
        if self._type == "numeric":
            return self._eval_numeric(x)["value"]
        else:
            return self._eval_new_data_categoric(x)

    def _eval_new_data_categoric(self, x):
        if self.is_response:
            return np.atleast_2d(np.where(x == self.data["reference"], 1, 0)).T
        else:
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
