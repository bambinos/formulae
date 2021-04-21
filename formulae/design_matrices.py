# pylint: disable=relative-beyond-top-level
import itertools
import logging

from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy as sp

from scipy import linalg

from .eval import EvalEnvironment
from .terms import Model, Intercept
from .model_description import model_description
from .utils import flatten_list

_log = logging.getLogger("formulae")


class DesignMatrices:
    """A wrapper of ResponseVector, CommonEffectsMatrix and GroupEffectsMatrix

    Parameters
    ----------

    model : Model
        The model description.
    data: pandas.DataFrame
        The data frame where variables are taken from
    eval_env: EvalEnvironment
        The evaluation environment object where we take values and functions from.
    """

    def __init__(self, model, data, eval_env):
        self.data = data
        self.eval_env = eval_env
        self.response = None
        self.common = None
        self.group = None
        self.model = model

        if self.model.response:
            self.response = ResponseVector(self.model.response)
            self.response._evaluate(data, eval_env)

        if self.model.common_terms:
            self.common = CommonEffectsMatrix(Model(*self.model.common_terms))
            self.common._evaluate(data, eval_env)

        if self.model.group_terms:
            self.group = GroupEffectsMatrix(self.model.group_terms)
            self.group._evaluate(data, eval_env)


class ResponseVector:
    """Representation of the respose vector of a model

    Parameters
    ----------

    term : Response
        The description and data of the response term.
    data: pandas.DataFrame
        The data frame where variables are taken from
    eval_env: EvalEnvironment
        The evaluation environment object where we take values and functions from.
    """

    def __init__(self, term):
        self.term = term
        self.data = None
        self.eval_env = None
        self.design_vector = None
        self.name = None  # a string
        self.type = None  # either numeric or categorical
        self.refclass = None  # Not None for categorical variables

    def _evaluate(self, data, eval_env):
        """Evaluates `self.term` inside the data mask provided by `data` and
        updates `self.design_vector` and `self.name`
        """
        self.data = data
        self.eval_env = eval_env
        self.term.set_type(self.data, self.eval_env)
        self.term.set_data()
        self.name = self.term.term.name
        self.design_vector = self.term.term.data
        self.type = self.term.term.metadata["type"]
        if self.type == "categoric":
            self.refclass = self.term.term.metadata["reference"]

    def as_dataframe(self):
        """Returns `self.design_vector` as a pandas.DataFrame"""
        data = pd.DataFrame(self.design_vector)
        if self.type == "categoric":
            colname = f"{self.name}[{self.refclass}]"
        else:
            colname = self.name
        data.columns = [colname]
        return data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "name=" + self.name,
            "type=" + self.type,
            "length=" + str(len(self.design_vector)),
        ]
        if self.type == "categoric":
            string_list += ["refclass=" + self.refclass]
        return "ResponseVector(" + ", ".join(string_list) + ")"


class CommonEffectsMatrix:
    """Representation of the design matrix for the common effects of a model.

    Parameters
    ----------

    model : Model
        An Model object containing terms for the common effects of the model.
    data: pandas.DataFrame
        The data frame where variables are taken from
    eval_env: EvalEnvironment
        The evaluation environment object where we take values and functions from.
    """

    def __init__(self, model):
        self.model = model
        self.data = None
        self.eval_env = None
        self.design_matrix = None
        self.terms_info = None
        self.evaluated = False

    def _evaluate(self, data, eval_env):
        """Obtain design matrix for common effects.

        Evaluates ``self.model`` inside the data mask provided by ``data`` and updates
        ``self.design_matrix``. It also populates the dictionary ``self.terms_info`` with
        information related to each term, such as the type, the columns they occupy in the design
        matrix and the names of the columns.
        """
        self.data = data
        self.eval_env = eval_env
        d = self.model.eval(self.data, self.eval_env)
        self.design_matrix = np.column_stack([d[key] for key in d.keys()])
        self.terms_info = {}
        # Get types and column slices
        start = 0
        for term in self.model.terms:
            self.terms_info[term.name] = term.metadata
            delta = d[term.name].shape[1]
            self.terms_info[term.name]["cols"] = slice(start, start + delta)
            self.terms_info[term.name]["full_names"] = self._term_full_names(term.name)
            start += delta

        self.evaluated = True

    def _evaluate_new_data(self, data):
        # Create and return new CommonEffectsMatrix from the information in the terms,
        # with the new data
        if not self.evaluated:
            raise ValueError("Can't evaluate new data on unevaluated matrix.")
        new_instance = self.__class__(self.model)
        new_instance.data = data
        new_instance.eval_env = self.eval_env
        new_instance.terms_info = deepcopy(self.terms_info)
        new_instance.design_matrix = np.column_stack(
            [term.eval_new_data(data) for term in self.model.terms]
        )
        new_instance.evaluated = True
        return new_instance

    def as_dataframe(self):
        """Returns `self.design_matrix` as a pandas.DataFrame"""
        colnames = [self._term_full_names(name) for name in self.terms_info]
        data = pd.DataFrame(self.design_matrix)
        data.columns = list(flatten_list(colnames))
        return data

    def _term_full_names(self, name):  # pylint: disable=inconsistent-return-statements
        # Always returns a list
        term = self.terms_info[name]
        _type = term["type"]
        if _type == "intercept":
            return ["Intercept"]
        elif _type == "numeric":
            return [name]
        elif _type == "interaction":
            return interaction_label(term)
        elif _type == "categoric":
            # "levels" is present when we have dummy encoding (not just a vector of 0-1)
            if "levels" in term.keys():
                # Ask if encoding is "full" or "reduced"
                if term["encoding"] == "full":
                    return [f"{name}[{level}]" for level in term["levels"]]
                else:
                    return [f"{name}[{level}]" for level in term["levels"][1:]]
            else:
                return [f"{name}[{term['reference']}]"]

    def __getitem__(self, term):
        if term not in self.terms_info.keys():
            raise ValueError(f"'{term}' is not a valid term name")
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
    """Representation of the design matrix for the group specific effects of a model.

    In this case, `self.design_matrix` is a sparse matrix in CSC format.
    The sub-matrix that corresponds to a specific group effect can be accessed by
    `self['1|g']`.

    Parameters
    ----------

    terms : list
        A list of GroupSpecificTerm objects.
    data: pandas.DataFrame
        The data frame where variables are taken from
    eval_env: EvalEnvironment
        The evaluation environment object where we take values and functions from.
    """

    def __init__(self, terms):
        self.terms = terms
        self.data = None
        self.eval_env = None
        self.design_matrix = np.zeros((0, 0))
        self.terms_info = {}
        self.evaluated = False

    def _evaluate(self, data, eval_env):
        """Evaluates `self.terms` inside the data mask provided by `data` and
        updates `self.design_matrix`.
        """
        self.data = data
        self.eval_env = eval_env
        start_row = 0
        start_col = 0
        Z = []
        self.terms_info = {}
        for term in self.terms:
            encoding = True
            # If both (1|g) and (x|g) are in the model, then the encoding for x is False.
            if not isinstance(term.expr, Intercept):
                for term_ in self.terms:
                    if term_.factor == term.factor and isinstance(term_.expr, Intercept):
                        encoding = False
            d = term.eval(self.data, self.eval_env, encoding)

            if d["type"] == "categoric":
                levels = d["levels"] if d["encoding"] == "full" else d["levels"][1:]
                for idx, level in enumerate(levels):
                    Xi = np.atleast_2d(d["Xi"][:, idx]).T
                    Ji = d["Ji"]
                    Zi = linalg.khatri_rao(Ji.T, Xi.T).T
                    delta_row, delta_col = Zi.shape
                    Z.append(Zi)
                    term_name = term.to_string(level)
                    self.terms_info[term_name] = {
                        "type": "categoric",
                        "Xi": Xi,
                        "Ji": Ji,
                        "groups": d["groups"],
                        "encoding": d["encoding"],
                        "levels": d["levels"],
                        "reference": d["reference"],
                        "full_names": [f"{term_name}[{group}]" for group in d["groups"]],
                    }
                    self.terms_info[term_name]["idxs"] = (
                        slice(start_row, start_row + delta_row),
                        slice(start_col, start_col + delta_col),
                    )
                    start_row += delta_row
                    start_col += delta_col
            else:
                Zi = d["Zi"]
                delta_row, delta_col = Zi.shape
                Z.append(Zi)
                term_name = term.to_string()
                self.terms_info[term_name] = {k: v for k, v in d.items() if k != "Zi"}
                self.terms_info[term_name]["idxs"] = (
                    slice(start_row, start_row + delta_row),
                    slice(start_col, start_col + delta_col),
                )
                self.terms_info[term_name]["full_names"] = self._term_full_names(term_name)
                start_row += delta_row
                start_col += delta_col

        # Stored in Compressed Sparse Column format
        if Z:
            self.design_matrix = sp.sparse.block_diag(Z).tocsc()

        self.evaluated = True

    def _evaluate_new_data(self, data):
        if not self.evaluated:
            raise ValueError("Can't evaluate new data on unevaluated matrix.")

        new_instance = self.__class__(self.terms)

        start_row = start_col = 0
        Z = []

        for term in self.terms:
            d = term.eval_new_data(data)
            if d["type"] == "categoric":
                levels = d["levels"] if d["encoding"] == "full" else d["levels"][1:]
                Ji = d["Ji"]
                for idx, level in enumerate(levels):
                    Xi = np.atleast_2d(d["Xi"][:, idx]).T
                    Zi = linalg.khatri_rao(Ji.T, Xi.T).T
                    delta_row, delta_col = Zi.shape
                    Z.append(Zi)
                    term_name = term.to_string(level)
                    # All the info, except from the indexes, is copied.
                    new_instance.terms_info[term_name] = deepcopy(self.terms_info[term_name])
                    new_instance.terms_info[term_name]["idxs"] = (
                        slice(start_row, start_row + delta_row),
                        slice(start_col, start_col + delta_col),
                    )
                    start_row += delta_row
                    start_col += delta_col
            else:
                Zi = d["Zi"]
                delta_row, delta_col = Zi.shape
                Z.append(Zi)
                term_name = term.to_string()
                new_instance.terms_info[term_name] = deepcopy(self.terms_info[term_name])
                new_instance.terms_info[term_name]["idxs"] = (
                    slice(start_row, start_row + delta_row),
                    slice(start_col, start_col + delta_col),
                )
                start_row += delta_row
                start_col += delta_col

        new_instance.data = data
        new_instance.eval_env = self.eval_env

        # Stored in Compressed Sparse Column format
        if Z:
            new_instance.design_matrix = sp.sparse.block_diag(Z).tocsc()

        return new_instance

    def _term_full_names(self, name):  # pylint: disable=inconsistent-return-statements
        # Always returns a list
        term = self.terms_info[name]
        _type = term["type"]
        if _type in ["intercept", "numeric"]:
            return [f"{name}[{group}]" for group in term["groups"]]
        elif _type == "interaction":
            return interaction_label(term)
        elif _type == "categoric":
            # not used anymore?
            if "levels" in term.keys():
                # Ask if encoding is "full" or "reduced"
                if term["encoding"] == "full":
                    products = product(term["levels"], term["groups"])
                else:
                    products = product(term["levels"][1:], term["groups"])
            else:
                products = product([term["reference"]], term["groups"])
            return [f"{name}[{p[0]}|{p[1]}]" for p in products]

    def __getitem__(self, term):
        if term not in self.terms_info.keys():
            raise ValueError(f"'{term}' is not a valid term name")
        return self.design_matrix[self.terms_info[term]["idxs"]].toarray()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        terms_str = ",\n  ".join([f"'{k}': {{{term_str(v)}}}" for k, v in self.terms_info.items()])
        string_list = [
            "shape: " + str(self.design_matrix.shape),
            "terms: {\n    " + "  ".join(terms_str.splitlines(True)) + "\n  }",
        ]
        return "GroupEffectsMatrix(\n  " + ",\n  ".join(string_list) + "\n)"


def design_matrices(formula, data, na_action="drop", eval_env=0):
    """Obtain design matrices.

    Parameters
    ----------
    formula : string
        A model description written in the formula language
    data: pandas.DataFrame
        The data frame where variables are taken from
    na_action: string
        Describes what to do with missing values in `data`. 'drop' means to drop
        all rows with a missing value, 'error' means to raise an error. Defaults
        to 'drop'.
    eval_env: integer
        The number of environments we walk up in the stack starting from the function's caller
        to capture the environment where formula is evaluated. Defaults to 0 which means
        the evaluation environment is the environment where `design_matrices` is called.

    Returns
    ----------
    design: DesignMatrices
        An instance of DesignMatrices that contains the design matrice(s) described by
        `formula`.
    """

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

    eval_env = EvalEnvironment.capture(eval_env, reference=1)

    description = model_description(formula)

    # Incomplete rows are calculated using columns involved in model formula only
    cols_to_select = description.var_names.intersection(set(data.columns))
    data = data[list(cols_to_select)]

    incomplete_rows = data.isna().any(axis=1)
    incomplete_rows_n = incomplete_rows.sum()

    if incomplete_rows_n > 0:
        if na_action == "drop":
            _log.info(
                "Automatically removing %s/%s rows from the dataset.",
                incomplete_rows_n,
                data.shape[0],
            )
            data = data[~incomplete_rows]
        else:
            raise ValueError(f"'data' contains {incomplete_rows_n} incomplete rows.")

    design = DesignMatrices(description, data, eval_env)
    return design


# Utils
def term_str(term):
    x = None
    if term["type"] == "interaction":
        terms = term["terms"]
        vars_ = []
        for k, v in terms.items():
            if v["type"] == "numeric":
                vars_.append(f"    {k}: {{type={v['type']}}}")
            elif v["type"] == "categoric":
                str_l = [k2 + "=" + str(v2) for k2, v2 in v.items() if k2 != "value"]
                vars_.append(f"    {k}: {{" + ", ".join(str_l) + "}")
        x = f"type=interaction, cols={term['cols']}, full_names={term['full_names']}"
        x += ",\n    vars={\n  " + ",\n  ".join(vars_) + "\n  }"
    else:
        x = ", ".join([k + "=" + str(v) for k, v in term.items() if k not in ["Xi", "Ji"]])
    return x


def interaction_label(x):
    terms = x["terms"]
    colnames = []

    for k, v in terms.items():
        if v["type"] == "numeric":
            colnames.append([k])
        if v["type"] == "categoric":
            if "levels" in v.keys():
                # ask whether encoding is full or reduced
                if v["encoding"] == "full":
                    colnames.append([f"{k}[{level}]" for level in v["levels"]])
                else:
                    colnames.append([f"{k}[{level}]" for level in v["levels"][1:]])
            else:
                colnames.append([f"{k}[{v['reference']}]"])

    return [":".join(str_tuple) for str_tuple in list(itertools.product(*colnames))]
