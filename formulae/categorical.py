import numpy as np


class ContrastMatrix:
    def __init__(self, matrix, labels):
        if matrix.shape[1] != len(labels):
            raise ValueError(
                "The number of columns in the contrast matrix differs from the number of labels!"
            )
        self.matrix = matrix
        self.labels = labels


class CategoricalBox:
    def __init__(self, data, contrast, levels):
        self.data = data
        self.contrast = contrast
        self.levels = levels


class Treatment:
    def __init__(self, reference=None):
        """reference is the value of the reference itself"""
        self.reference = reference

    def code_with_intercept(self, levels):
        """This contrast matrix spans the intercept"""
        contrast = np.eye(len(levels))
        return ContrastMatrix(contrast, levels)

    def code_without_intercept(self, levels):
        """This contrast matrix _does not_ spans the intercept"""

        # First category is the default reference
        if self.reference is None:
            reference = 0
        else:
            if self.reference in levels:
                reference = levels.index(self.reference)
            else:
                raise ValueError("reference not in levels")

        eye = np.eye(len(levels) - 1)
        contrast = np.vstack(
            (eye[:reference, :], np.zeros((1, len(levels) - 1)), eye[reference:, :])
        )
        levels = levels[:reference] + levels[reference + 1 :]
        labels = [f"T.{level}" for level in levels]
        return ContrastMatrix(contrast, labels)


class Sum:
    def __init__(self, omit=None):
        """
        Compares the mean of each level to the mean-of-means.
        In a balanced design, compares the mean of each level to the overall mean.

        For full-rank coding, a standard intercept term is added.
        This intercept represents the mean of the variable.

        One level must be omitted to avoid redundancy. By default, this is the last level, but this
        can be adjusted via the `omit` argument.
        """
        self.omit = omit

    def _omit_index(self, levels):
        """Returns a number between 0 and len(levels) - 1"""
        if self.omit is None:
            # This assumes we use the last one as default...
            return len(levels) - 1
        else:
            return levels.index(self.omit)

    def _sum_contrast(self, levels):
        n = len(levels)
        omit_index = self._omit_index(levels)
        eye = np.eye(n - 1)
        out = np.empty((n, n - 1))

        out[:omit_index, :] = eye[:omit_index, :]
        out[omit_index, :] = -1
        out[omit_index + 1 :, :] = eye[omit_index:, :]
        return out

    def code_with_intercept(self, levels):
        contrast = self.code_without_intercept(levels)
        matrix = np.column_stack((np.ones(len(levels)), contrast.matrix))

        labels = ["mean"] + contrast.labels
        return ContrastMatrix(matrix, labels)

    def code_without_intercept(self, levels):
        matrix = self._sum_contrast(levels)
        omit_index = self._omit_index(levels)
        levels = levels[:omit_index] + levels[omit_index + 1 :]
        labels = [f"S.{level}" for level in levels]
        return ContrastMatrix(matrix, labels)


ENCODINGS = {"Treatment": Treatment, "Sum": Sum}
