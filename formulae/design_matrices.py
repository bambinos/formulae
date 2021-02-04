import logging

import numpy as np
import pandas as pd
import scipy as sp

from .terms import ModelTerms
from .model_description import model_description

_log = logging.getLogger("formulae")


class DesignMatrices:
    """Wraps ResponseVector CommonEffectsMatrix and GroupEffectsMatrix

    Parameters
    ----------

    model : ModelTerms
        The model description.
    data: DataFrame or dict
        The data object where we take values from.
    """

    def __init__(self, model, data):
        self.response = None
        self.common = None
        self.group = None
        self.model = model

        if self.model.response is not None:
            self.response = ResponseVector(self.model.response, data)

        if self.model.common_terms:
            self.common = CommonEffectsMatrix(ModelTerms(*self.model.common_terms), data)

        if self.model.group_terms:
            self.group = GroupEffectsMatrix(self.model.group_terms, data)


class ResponseVector:
    """Representation of the respose vector of a model"""

    def __init__(self, term, data):
        self.name = None  # a string
        self.data = None  # 1d numpy array
        self.type = None  # either numeric or categorical
        self.refclass = None  # Not None for categorical variables
        self.term = term
        self.evaluate(data)

    def evaluate(self, data):
        """Evaluates `self.term` inside the data mask provided by `data` and
        updates `self.y` and `self.name`
        """
        d = self.term.eval(data)
        self.data = d["value"]
        self.type = d["type"]
        if self.type == "categoric":
            self.refclass = d["reference"]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "name=" + self.term.term.name,
            "type=" + self.type,
            "length=" + str(len(self.data)),
        ]
        if self.type == "categoric":
            string_list += ["refclass=" + self.refclass]
        return "ResponseVector(" + ", ".join(string_list) + ")"


class CommonEffectsMatrix:
    """Representation of the design matrix for the common effects of a model."""

    def __init__(self, terms, data):
        self.design_matrix = None
        self.terms_info = None
        self.terms = terms
        self.evaluate(data)

    def evaluate(self, data):
        d = self.terms.eval(data)
        self.design_matrix = np.column_stack([d[key]["value"] for key in d.keys()])
        self.terms_info = {}
        # Get types and column slices
        start = 0
        for key in d.keys():
            delta = d[key]["value"].shape[1]
            self.terms_info[key] = {"type": d[key]["type"], "cols": slice(start, start + delta)}
            if d[key]["type"] == "categoric":
                self.terms_info[key]["levels"] = d[key]["levels"]
                self.terms_info[key]["reference"] = d[key]["reference"]
            start += delta

    def __getitem__(self, term):
        if term not in self.terms_info.keys():
            raise ValueError(f"'{term}' is not a valid term name")
        else:
            return self.design_matrix[:, self.terms_info[term]["cols"]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        terms_str = ",\n  ".join([f"'{k}': {{{term_str(v)}}}" for k, v in self.terms_info.items()])
        string_list = [
            "shape: " + str(self.design_matrix.shape),
            "terms: {\n    " + "  ".join(terms_str.splitlines(True)) + "\n  }",
        ]
        return "CommonEffectsMatrix(\n  " + ",\n  ".join(string_list) + "\n)"


class GroupEffectsMatrix:
    """Representation of the design matrix for the group specific effects of a model."""

    def __init__(self, terms, data):
        self._design_matrix = None
        self.terms_info = None
        self.terms = terms
        self.evaluate(data)

    def evaluate(self, data):
        start_row = 0
        start_col = 0
        Z = []
        self.terms_info = {}
        for term in self.terms:
            d = term.eval(data)
            Zi = d["Zi"]
            delta_row = Zi.shape[0]
            delta_col = Zi.shape[1]
            Z.append(Zi)
            self.terms_info[term.to_string()] = {
                "type": d["type"],
                "idxs": (
                    slice(start_row, start_row + delta_row),
                    slice(start_col, start_col + delta_col),
                ),
            }
            if d["type"] == "categoric":
                self.terms_info[term.to_string()]["levels"] = d["levels"]
                self.terms_info[term.to_string()]["reference"] = d["reference"]
            start_row += delta_row
            start_col += delta_col
        self._design_matrix = sp.sparse.block_diag(Z)

    @property
    def design_matrix(self):
        return self._design_matrix.toarray()

    def __getitem__(self, term):
        if term not in self.terms_info.keys():
            raise ValueError(f"'{term}' is not a valid term name")
        else:
            return self.design_matrix[self.terms_info[term]["idxs"]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        terms_str = ",\n  ".join([f"'{k}': {{{term_str(v)}}}" for k, v in self.terms_info.items()])
        string_list = [
            "shape: " + str(self.design_matrix.shape),
            "terms: {\n    " + "  ".join(terms_str.splitlines(True)) + "\n  }",
        ]
        return "GroupEffectsMatrix(\n  " + ",\n  ".join(string_list) + "\n)"


def term_str(term):
    return ", ".join([k + "=" + str(v) for k, v in term.items()])


def design_matrices(formula, data, na_action="drop"):

    if not isinstance(formula, str):
        raise ValueError("'formula' must be a string.")

    if len(formula) == 0:
        raise ValueError("'formula' cannot be an empty string.")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("'data' must be a pandas.DataFrame.")

    if data.shape[0] == 0:
        raise ValueError("'data' does not contain any observation.")

    if data.shape[1] == 0:
        raise ValueError("'data' does not contain any variable.")

    if na_action not in ["drop", "error"]:
        raise ValueError("'na_action' must be either 'drop' or 'error'")

    # Model description is obtained to check variables and subset data.
    description = model_description(formula)
    formula_vars = description.vars

    if not formula_vars <= set(data.columns):
        column_diff = list(formula_vars - set(data.columns))
        raise ValueError(f"Variable(s) {', '.join(column_diff)} are not in 'data'")

    data = data[list(formula_vars)]

    incomplete_rows = data.isna().any(axis=1)
    incomplete_rows_n = incomplete_rows.sum()

    if incomplete_rows_n > 0:
        if na_action == "drop":
            _log.info(
                f"Automatically removing {incomplete_rows_n}/{data.shape[0]} rows from the dataset."
            )
            data = data[~incomplete_rows]
        else:
            raise ValueError(f"'data' contains {incomplete_rows_n} incomplete rows.")

    return DesignMatrices(model_description(formula), data)
