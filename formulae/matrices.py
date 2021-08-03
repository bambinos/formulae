# pylint: disable=relative-beyond-top-level
import itertools
import logging
import textwrap

from copy import deepcopy

import numpy as np
import pandas as pd

from .eval import EvalEnvironment
from .terms import Model, Intercept
from .model_description import model_description
from .utils import flatten_list

_log = logging.getLogger("formulae")


class DesignMatrices:
    """A wrapper of the response, the common and group specific effects.

    Parameters
    ----------

    model : Model
        The model description, the result of calling ``model_description``.
    data: pandas.DataFrame
        The data frame where variables are taken from.
    eval_env: EvalEnvironment
        The environment where values and functions are taken from.

    Attributes
    ----------
    response: ResponseVector
        The response in the model. Access its values with ``self.response.design_vector``. It is
        ``None`` if there is no response term in the model.
    common: CommonEffectsMatrix
        The common effects (a.k.a. fixed effects) in the model. The design matrix can be accessed
        with ``self.common.design_matrix``. The submatrix for a term is accessed via
        ``self.common[term_name]``. It is ``None`` if there are no common terms in the
        model.
    group: GroupEffectsMatrix
        The group specific effects (a.k.a. random effects) in the model. The design matrix can be
        accessed with ``self.group.design_matrix``. The submatrix for a term is accessed via
        ``self.group[term_name]``. It is ``None`` if there are no group specific terms in the
        model.
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
    """Representation of the respose vector of a model.

    Parameters
    ----------

    term : Response
        The term that represents the response in the model.
    data: pandas.DataFrame
        The data frame where variables are taken from.
    eval_env: EvalEnvironment
        The environment where values and functions are taken from.

    Attributes
    ----------
    design_vector: np.array
        A 1-dimensional numpy array containing the values of the response.
    name: string
        The name of the response term.
    type: string
        Either ``"numeric"`` or ``"categoric"``.
    baseline: string
        The name of the class taken as reference if ``type = "categoric"``.
    """

    def __init__(self, term):
        self.term = term
        self.data = None
        self.eval_env = None
        self.design_vector = None
        self.name = None  # a string
        self.type = None  # either numeric or categorical
        self.baseline = None  # Not None for non-binary categorical variables
        self.success = None  # Not None for binary categorical variables
        self.levels = None  # Not None for categorical variables
        self.binary = None  # Not None for categorical variables (either True or False)

    def _evaluate(self, data, eval_env):
        """Evaluates ``self.term`` inside the data mask provided by ``data`` and
        updates ``self.design_vector`` and ``self.name``.
        """
        self.data = data
        self.eval_env = eval_env
        self.term.set_type(self.data, self.eval_env)
        self.term.set_data()
        self.name = self.term.term.name
        self.design_vector = self.term.term.data
        self.type = self.term.term.metadata["type"]

        if self.type == "categoric":
            self.binary = len(np.unique(self.design_vector)) == 2
            self.levels = self.term.term.metadata["levels"]
            if self.binary:
                self.success = self.term.term.metadata["reference"]
            else:
                self.baseline = self.term.term.metadata["reference"]

    def _evaluate_new_data(self, data):
        if self.type == "proportion":
            return self.term.term.eval_new_data(data)
        raise ValueError("Can't evaluate response term with type different to 'proportion'")

    def as_dataframe(self):
        """Returns ``self.design_vector`` as a pandas.DataFrame."""
        data = pd.DataFrame(self.design_vector)
        if self.type == "categoric":
            colname = f"{self.name}[{self.baseline}]"
        else:
            colname = self.name
        data.columns = [colname]
        return data

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            f"name: {self.name}",
            f"type: {self.type}",
            f"length: {len(self.design_vector)}",
        ]
        if self.type == "categoric":
            string_list += [f"levels: {self.levels}", f"binary: {self.binary}"]
            if self.binary:
                string_list += [f"success: {self.success}"]
            else:
                string_list += [f"baseline: {self.baseline}"]
        return f"ResponseVector({wrapify(spacify(multilinify(string_list)))}\n)"


class CommonEffectsMatrix:
    """Representation of the design matrix for the common effects of a model.

    Parameters
    ----------

    model : Model
        A ``Model`` object containing only terms for the common effects of the model.
    data: pandas.DataFrame
        The data frame where variables are taken from.
    eval_env: EvalEnvironment
        The environment where values and functions are taken from.

    Attributes
    ----------
    design_matrix: np.array
        A 2-dimensional numpy array containing the values of the design matrix.
    evaluated: bool
        Indicates if the terms have been evaluated at least once. The terms must have been evaluated
        before calling ``self._evaluate_new_data()`` because we must know the type of each term
        to correctly handle the new data passed and the terms here.
    terms_info: dict
        A dictionary that holds information related to each of the common specific terms, such as
        ``"cols"``, ``"type"``, and ``"full_names"``. If ``"type"`` is ``"categoric"``, it also
        contains ``"groups"``, ``"encoding"``, ``"levels"``, and ``"reference"``.
        The keys are given by the term names.
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
        ``self.design_matrix``. This method also sets the values of ``self.data`` and
        ``self.eval_env``.

        It also populates the dictionary ``self.terms_info`` with information related to each term,
        such as the type, the columns they occupy in the design matrix and the names of the columns.

        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from
        eval_env: EvalEnvironment
            The environment where values and functions are taken from.
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
            if term._type == "interaction":  # pylint: disable = protected-access
                self.terms_info[term.name]["levels"] = self._interaction_levels(term.name)
            self.terms_info[term.name]["full_names"] = self._term_full_names(term.name)
            self.terms_info[term.name]["cols"] = slice(start, start + delta)
            start += delta

        self.evaluated = True

    def _evaluate_new_data(self, data):
        """Evaluates common terms with new data and return a new instance of
        ``CommonEffectsMatrix``.

        This method is intended to be used to obtain design matrices for new data and obtain
        out of sample predictions. Stateful transformations are properly handled if present in any
        of the terms, which means parameters involved in the transformation are not overwritten with
        the new data.

        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from

        Returns
        ----------
        new_instance: CommonEffectsMatrix
            A new instance of ``CommonEffectsMatrix`` whose design matrix is obtained with the
            values in the new data set.
        """
        # Create and return new CommonEffectsMatrix from the information in the terms, with new data
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
        """Returns `self.design_matrix` as a pandas.DataFrame."""
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
        elif _type in ["numeric", "offset"]:
            return [name]
        elif _type == "interaction":
            return interaction_label(term)
        elif _type == "categoric":
            # "levels" is present when we have dummy encoding (not just a vector of 0-1)
            if "levels" in term.keys():
                # Ask if encoding is "full" or "reduced"
                levels = term["levels"] if term["encoding"] == "full" else term["levels"][1:]
                return [f"{name}[{level}]" for level in levels]
            else:
                return [f"{name}[{term['reference']}]"]

    def _interaction_levels(self, name):
        terms = self.terms_info[name]["terms"]
        colnames = []
        for v in terms.values():
            if v["type"] == "categoric":
                levels = v["levels"] if v["encoding"] == "full" else v["levels"][1:]
                colnames.append([str(level) for level in levels])
        if colnames:
            return [", ".join(str_tuple) for str_tuple in list(itertools.product(*colnames))]
        else:
            return None

    def __getitem__(self, term):
        """Get the sub-matrix that corresponds to a given term.

        Parameters
        ----------
        term: string
            The name of the term.

        Returns
        ----------
        matrix: np.array
            A 2-dimensional numpy array that represents the sub-matrix corresponding to the
            term passed.
        """
        if term not in self.terms_info.keys():
            raise ValueError(f"'{term}' is not a valid term name")
        return self.design_matrix[:, self.terms_info[term]["cols"]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = [f"'{k}': {{{spacify(term_str(v))}\n}}" for k, v in self.terms_info.items()]
        string = multilinify(string)
        string = [
            f"shape: {self.design_matrix.shape}",
            f"terms: {{{spacify(string)}\n}}",
        ]
        return f"CommonEffectsMatrix({wrapify(spacify(multilinify(string)))}\n)"


class GroupEffectsMatrix:
    """Representation of the design matrix for the group specific effects of a model.

    The sub-matrix that corresponds to a specific group effect can be accessed by
    ``self[term_name]``, for example ``self["1|g"]``.

    Parameters
    ----------
    terms : list
        A list of ``GroupSpecificTerm`` objects.
    data: pandas.DataFrame
        The data frame where variables are taken from
    eval_env: EvalEnvironment
        The environment where values and functions are taken from.

    Attributes
    ----------
    design_matrix: np.array
        A 2 dimensional numpy array with the values of the design matrix.
    evaluated: bool
        Indicates if the terms have been evaluated at least once. The terms must have been evaluated
        before calling ``self._evaluate_new_data()`` because we must know the type of each term
        to correctly handle the new data passed and the terms here.
    terms_info: dict
        A dictionary that holds information related to each of the group specific terms, such as
        the matrices ``"Xi"`` and ``"Ji"``, ``"cols"``, ``"type"``, and ``"full_names"``. If
        ``"type"`` is ``"categoric"``, it also contains ``"groups"``, ``"encoding"``, ``"levels"``,
        and ``"reference"``. The keys are given by the term names.
    """

    def __init__(self, terms):
        self.terms = terms
        self.data = None
        self.eval_env = None
        self.design_matrix = np.zeros((0, 0))
        self.terms_info = {}
        self.evaluated = False

    def _evaluate(self, data, eval_env):
        """Evaluate group specific terms.

        This evaluates ``self.terms`` inside the data mask provided by ``data`` and the environment
        ``eval_env``. It updates ``self.design_matrix`` with the result from the evaluation of each
        term.

        This method also sets the values of ``self.data`` and ``self.eval_env``. It also populates
        the dictionary ``self.terms_info`` with information related to each term,such as the type,
        the columns and rows they occupy in the design matrix and the names of the columns.

        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from
        eval_env: EvalEnvironment
            The environment where values and functions are taken from.
        """
        self.data = data
        self.eval_env = eval_env
        start = 0
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

            # Grab subcomponent of Z that corresponds to this term
            Zi = d["Zi"]
            delta = Zi.shape[1]
            Z.append(Zi)
            name = term.get_name()
            self.terms_info[name] = {k: v for k, v in d.items() if k != "Zi"}
            if self.terms_info[name]["type"] == "interaction":  # pylint: disable = protected-access
                self.terms_info[name]["levels"] = self._interaction_levels(name)
            # Generate term names
            self.terms_info[name]["full_names"] = self._term_full_names(name, term.expr.name)
            self.terms_info[name]["cols"] = slice(start, start + delta)
            start += delta

        if Z:
            self.design_matrix = np.column_stack(Z)
        self.evaluated = True

    def _evaluate_new_data(self, data):
        """Evaluates group specific terms with new data and return a new instance of
        ``GroupEffectsMatrix``.

        This method is intended to be used to obtain design matrices for new data and obtain
        out of sample predictions. Stateful transformations are properly handled if present in any
        of the group specific terms, which means parameters involved in the transformation are not
        overwritten with the new data.


        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from

        Returns
        ----------
        new_instance: GroupEffectsMatrix
            A new instance of ``GroupEffectsMatrix`` whose design matrix is obtained with the values
            in the new data set.
        """
        if not self.evaluated:
            raise ValueError("Can't evaluate new data on unevaluated matrix.")

        new_instance = self.__class__(self.terms)
        start = 0
        Z = []
        for term in self.terms:
            d = term.eval_new_data(data)
            # Grab subcomponent of Z that corresponds to this term
            Zi = d["Zi"]
            delta = Zi.shape[1]
            Z.append(Zi)
            name = term.get_name()
            new_instance.terms_info[name] = deepcopy(self.terms_info[name])
            new_instance.terms_info[name]["cols"] = slice(start, start + delta)
            start += delta
        new_instance.data = data
        new_instance.eval_env = self.eval_env
        if Z:
            new_instance.design_matrix = np.column_stack(Z)
        return new_instance

    def _term_full_names(self, name, expr):
        # Always returns a list. This should be clearer in the future.
        term = self.terms_info[name]
        groups = term["groups"]
        if term["type"] in ["intercept", "numeric"]:
            names = [f"{name}[{group}]" for group in groups]
        elif term["type"] == "interaction":
            levels = interaction_label(term)
            names = [f"{level}|{group}" for group in groups for level in levels]
        elif term["type"] == "categoric":
            if "levels" in term.keys():
                # Ask if encoding is "full" or "reduced"
                levels = term["levels"] if term["encoding"] == "full" else term["levels"][1:]
            else:
                levels = [term["reference"]]
            names = [f"{expr}[{level}]|{group}" for group in groups for level in levels]
        return names

    def _interaction_levels(self, name):
        terms = self.terms_info[name]["terms"]
        colnames = []
        for v in terms.values():
            if v["type"] == "categoric":
                levels = v["levels"] if v["encoding"] == "full" else v["levels"][1:]
                colnames.append([str(level) for level in levels])
        if colnames:
            return [", ".join(str_tuple) for str_tuple in list(itertools.product(*colnames))]
        else:
            return None

    def __getitem__(self, term):
        """Get the sub-matrix that corresponds to a given term.

        Parameters
        ----------
        term: string
            The name of a group specific term.

        Returns
        ----------
        matrix: np.array
            A 2-dimensional numpy array that represents the sub-matrix corresponding to the
            term passed.
        """
        if term not in self.terms_info.keys():
            raise ValueError(f"'{term}' is not a valid term name")
        return self.design_matrix[:, self.terms_info[term]["cols"]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = [f"'{k}': {{{spacify(term_str(v))}\n}}" for k, v in self.terms_info.items()]
        string = multilinify(string)
        string = [
            f"shape: {self.design_matrix.shape}",
            f"terms: {{{spacify(string)}\n}}",
        ]
        return f"GroupEffectsMatrix({wrapify(spacify(multilinify(string)))}\n)"


def design_matrices(formula, data, na_action="drop", eval_env=0):
    """Parse model formula and obtain a ``DesignMatrices`` object containing objects representing
    the response and the design matrices for both the common and group specific effects.

    Parameters
    ----------
    formula : string
        A model formula.
    data: pandas.DataFrame
        The data frame where variables in the formula are taken from.
    na_action: string
        Describes what to do with missing values in ``data``. ``"drop"`` means to drop
        all rows with a missing value, ``"error"`` means to raise an error. Defaults
        to ``"drop"``.
    eval_env: integer
        The number of environments we walk up in the stack starting from the function's caller
        to capture the environment where formula is evaluated. Defaults to 0 which means
        the evaluation environment is the environment where ``design_matrices`` is called.

    Returns
    ----------
    design: DesignMatrices
        An instance of DesignMatrices that contains the design matrice(s) described by
        ``formula``.
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
    if term["type"] == "interaction":
        string_list = [f"{k}: {v}" for k, v in term.items() if k not in ["terms", "Xi", "Ji"]]
        string_vars = [f"'{k}': {{{spacify(term_str(v))}\n}}" for k, v in term["terms"].items()]
        string = multilinify(string_list + [f"vars: {{{spacify(multilinify(string_vars))}\n}}"])
    else:
        string = multilinify([f"{k}: {v}" for k, v in term.items() if k not in ["Xi", "Ji"]])
    return string


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


def spacify(string):
    return "  " + "  ".join(string.splitlines(True))


def multilinify(l, sep=","):
    sep += "\n"
    return "\n" + sep.join(l)


def wrapify(string, width=100):
    l = string.splitlines(True)
    wrapper = textwrap.TextWrapper(width=width)
    for idx, line in enumerate(l):
        if len(line) > width:
            leading_spaces = len(line) - len(line.lstrip(" "))
            wrapper.subsequent_indent = " " * (leading_spaces + 2)
            wrapped = wrapper.wrap(line)
            l[idx] = "\n".join(wrapped) + "\n"
    return "".join(l)
