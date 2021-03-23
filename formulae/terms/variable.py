import numpy as np
import pandas as pd

from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

class Variable:
    """Atomic component of a Term"""

    def __init__(self, name, level=None):
        self.data = None
        self._intermediate_data = None
        self._type = None
        self.name = name
        self.level = level

    def __hash__(self):
        return hash((self._type, self.name, self.level))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._type == other.type_ and self.name == other.name and self.level == other.level

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, level='{self.level}')"

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

    def set_data(self, encoding=None, is_response=False):
        """Obtains and stores the final data object related to this variable.

        Evaluates the variable according to its type and stores the result in ``.data_mask``. It
        does not support multi-level categoric responses yet, it is, If ``is_response`` is ``True``
        and the variable is of a categoric type, this method returns a 1d array of 0-1 instead of a
        matrix.
        """
        if self._type is None:
            raise ValueError("Variable type is not set.")
        if self._type not in ["numeric", "categoric"]:
            raise ValueError(f"Variable is of an unrecognized type ({self._type}).")
        if self._type == "numeric":
            self.data = self._eval_numeric(self._intermediate_data)
        elif self._type == "categoric":
            self.data = self._eval_categoric(self._intermediate_data, encoding, is_response)
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
        out = {"value": value, "type": "numeric"}
        return out

    def _eval_categoric(self, x, encoding, is_response):
        # If not ordered, we make it ordered.
        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            categories = sorted(x.unique().tolist())
            cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if is_response:
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

    def eval_new_data(self, data_mask, encoding=None, is_response=False):
        """Evaluates the function call with new data.

        This method evaluates the function call within a new data mask. If the transformation
        applied is a stateful transformation, it uses the proper object that remembers all
        parameters or settings that may have been set in a first pass.
        """
        x = data_mask[self.name]
        if self._type == "numeric":
            return self._eval_numeric(x)["value"]
        else:
            # TODO: This doesn't work, or shouldn't work because we should remember
            # encoding and all that.
            return self._eval_categoric(x, encoding, is_response)["value"]