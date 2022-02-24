from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ContrastMatrix:
    """A representation of a contrast matrix

    Parameters
    ----------
    contrast: 2-dimensional np.array
        The contrast matrix as a numpy array.
    labels: list or tuple
        The labels for the columns of the contrast matrix. Its length must match the number of
        columns in the contrast matrix.
    """

    def __init__(self, matrix, labels):
        self.matrix = matrix
        self.labels = labels
        if matrix.shape[1] != len(labels):  # pragma: no cover
            raise ValueError(
                "The number of columns in the contrast matrix is not equal to the number of labels"
            )

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if not (
            isinstance(value, np.ndarray) and value.dtype.kind in "if" and value.ndim == 2
        ):  # pragma: no cover
            raise ValueError("The matrix argument must be a 2d numerical numpy array")
        self._matrix = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if not isinstance(value, (list, tuple)):  # pragma: no cover
            raise ValueError("The labels argument must be a list or a tuple")

        if not all(isinstance(i, str) for i in value):  # pragma: no cover
            raise ValueError("The items in the labels argument must be of type 'str'")

        self._labels = value

    def __str__(self):  # pragma: no cover
        msg = (
            f"{self.__class__.__name__}\n"
            f"Matrix:\n{self.matrix}\n\n"
            f"Labels:\n{', '.join(self.labels)}"
        )
        return msg

    def __repr__(self):  # pragma: no cover
        return self.__str__()


class CategoricalBox:
    """A container with information to encode categorical data

    Parameters
    ----------
    data: 1d array-like
        The data converted to categorical.
    contrast: Encoding
        An instance that represents the contrast matrix used to encode the categorical variable.
    levels: list or tuple
        The levels in ``data`` in the desired order.
    """

    def __init__(self, data, contrast, levels):
        self.data = data
        self.contrast = contrast
        self.levels = levels

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, pd.Series):
            value = value.values
        if not (isinstance(value, np.ndarray) and value.ndim == 1):  # pragma: no cover
            raise ValueError("The data argument must be one dimensional array-like")
        self._data = value

    @property
    def contrast(self):
        return self._contrast

    @contrast.setter
    def contrast(self, value):
        # Allows to do C(x, Treatment) instead of C(x, Treatment())
        if callable(value):
            value = value()

        if not (isinstance(value, Encoding) or value is None):  # pragma: no cover
            raise ValueError("The contrast argument in must be an instance of Encoding")
        self._contrast = value

    @property
    def levels(self):
        return self._levels

    @levels.setter
    def levels(self, value):
        if value is not None and set(value) != set(self.data):  # pragma: no cover
            raise ValueError("The levels beign assigned and the levels in the data differ")
        self._levels = value


class Encoding(ABC):
    """Abstract class for custom Encodings"""

    @abstractmethod
    def code_with_intercept(self, levels):  # pragma: no cover
        """This contrast matrix _does_ span the intercept"""
        return

    @abstractmethod
    def code_without_intercept(self, levels):  # pragma: no cover
        """This contrast matrix _does not_ span the intercept"""
        return


class Treatment(Encoding):
    def __init__(self, reference=None):
        """Treatment encoding

        This is also known as dummy encoding.

        When the encoding is not full-rank, one level is taken as reference and the regression
        coefficients measure the difference between the each level and the reference. In that case,
        the intercept represents the mean of the reference level.

        When the encoding is of full-rank, there's a dummy variable for each level representing
        its mean.

        Parameters
        ----------
        reference: str
            The level to take as reference
        """
        self.reference = reference

    def code_with_intercept(self, levels):
        contrast = np.eye(len(levels), dtype=int)
        labels = [str(level) for level in levels]
        return ContrastMatrix(contrast, labels)

    def code_without_intercept(self, levels):
        # First category is the default reference
        if self.reference is None:
            reference = 0
        else:
            if self.reference in levels:
                reference = levels.index(self.reference)
            else:  # pragma: no cover
                raise ValueError("reference not in levels")

        eye = np.eye(len(levels) - 1, dtype=int)
        contrast = np.vstack(
            (eye[:reference, :], np.zeros((1, len(levels) - 1)), eye[reference:, :])
        )
        levels = levels[:reference] + levels[reference + 1 :]
        labels = [str(level) for level in levels]
        return ContrastMatrix(contrast, labels)


class Sum(Encoding):
    def __init__(self, omit=None):
        """Sum-to-zero encoding

        This is also known as deviation encoding. It compares the the mean of each level to the
        grand mean (aka mean-of-means).

        For full-rank coding, an intercept term is added. This intercept represents the mean
        of the response variable.

        One level must be omitted to avoid redundancy. By default, this is the last level, but this
        can be adjusted via the `omit` argument.

        Parameters
        ----------
        omit: str
            The level to omit.
        """
        self.omit = omit

    def _omit_index(self, levels):
        """Returns a number between 0 and len(levels) - 1"""
        if self.omit is None:
            # By default, omit the lats level.
            return len(levels) - 1
        else:
            return levels.index(self.omit)

    def _sum_contrast(self, levels):
        n = len(levels)
        omit_index = self._omit_index(levels)
        eye = np.eye(n - 1, dtype=int)
        out = np.empty((n, n - 1), dtype=int)

        out[:omit_index, :] = eye[:omit_index, :]
        out[omit_index, :] = -1
        out[omit_index + 1 :, :] = eye[omit_index:, :]
        return out

    def code_with_intercept(self, levels):
        contrast = self.code_without_intercept(levels)
        matrix = np.column_stack((np.ones(len(levels), dtype=int), contrast.matrix))

        labels = ["mean"] + contrast.labels
        return ContrastMatrix(matrix, labels)

    def code_without_intercept(self, levels):
        matrix = self._sum_contrast(levels)
        omit_index = self._omit_index(levels)
        levels = levels[:omit_index] + levels[omit_index + 1 :]
        labels = [str(level) for level in levels]
        return ContrastMatrix(matrix, labels)


ENCODINGS = {"Treatment": Treatment, "Sum": Sum}
