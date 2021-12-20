# pylint: disable=relative-beyond-top-level
import logging
import textwrap

import numpy as np
import pandas as pd

from .environment import Environment
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
    env: Environment
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

    def __init__(self, model, data, env):
        self.data = data
        self.env = env
        self.response = None
        self.common = None
        self.group = None
        self.model = model

        # Evaluate terms in the model
        self.model.eval(data, env)

        if self.model.response:
            self.response = ResponseVector(self.model.response)
            self.response._evaluate(data, env)

        if self.model.common_terms:
            self.common = CommonEffectsMatrix(self.model.common_terms)
            self.common._evaluate(data, env)

        if self.model.group_terms:
            self.group = GroupEffectsMatrix(self.model.group_terms)
            self.group._evaluate(data, env)


class ResponseVector:
    """Representation of the respose vector of a model.

    Parameters
    ----------

    term : Response
        The term that represents the response in the model.
    data: pandas.DataFrame
        The data frame where variables are taken from.
    env: Environment
        The environment where values and functions are taken from.

    Attributes
    ----------
    design_vector: np.array
        A 1-dimensional numpy array containing the values of the response.
    name: string
        The name of the response term.
    kind: string
        Either ``"numeric"`` or ``"categoric"``.
    baseline: string
        The name of the class taken as reference if ``kind = "categoric"``.
    """

    def __init__(self, term):
        self.term = term
        self.data = None
        self.env = None
        self.design_vector = None
        self.name = None  # a string
        self.kind = None  # either numeric or categorical
        self.baseline = None  # Not None for non-binary categorical variables
        self.success = None  # Not None for binary categorical variables
        self.levels = None  # Not None for categorical variables
        self.binary = None  # Not None for categorical variables (either True or False)

    def _evaluate(self, data, env):
        """Evaluates ``self.term`` inside the data mask provided by ``data`` and
        updates ``self.design_vector`` and ``self.name``.
        """
        self.data = data
        self.env = env
        self.term.set_type(self.data, self.env)
        self.term.set_data()
        self.name = self.term.term.name
        self.design_vector = self.term.term.data
        self.kind = self.term.term.metadata["kind"]

        if self.kind == "categoric":
            self.binary = len(np.unique(self.design_vector)) == 2
            self.levels = self.term.term.metadata["levels"]
            if self.binary:
                self.success = self.term.term.metadata["reference"]
            else:
                self.baseline = self.term.term.metadata["reference"]

    def _evaluate_new_data(self, data):
        if self.kind == "proportion":
            return self.term.term.eval_new_data(data)
        raise ValueError("Can't evaluate response term with kind different to 'proportion'")

    def as_dataframe(self):
        """Returns ``self.design_vector`` as a pandas.DataFrame."""
        data = pd.DataFrame(self.design_vector)
        if self.kind == "categoric":
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
            f"kind: {self.kind}",
            f"length: {len(self.design_vector)}",
        ]
        if self.kind == "categoric":
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
    env: Environment
        The environment where values and functions are taken from.

    Attributes
    ----------
    design_matrix: np.array
        A 2-dimensional numpy array containing the values of the design matrix.
    evaluated: bool
        Indicates if the terms have been evaluated at least once. The terms must have been evaluated
        before calling ``self._evaluate_new_data()`` because we must know the kind of each term
        to correctly handle the new data passed and the terms here.
    terms_info: dict
        A dictionary that holds information related to each of the common specific terms, such as
        ``"cols"``, ``"kind"``, and ``"labels"``. If ``"kind"`` is ``"categoric"``, it also
        contains ``"groups"``, ``"encoding"``, ``"levels"``, and ``"reference"``.
        The keys are given by the term names.
    """

    def __init__(self, terms):
        self.terms = terms
        self.data = None
        self.env = None
        self.design_matrix = None
        self.evaluated = False
        self.slices = {}

    def _evaluate(self, data, env):
        """Obtain design matrix for common effects.

        Evaluates ``self.model`` inside the data mask provided by ``data`` and updates
        ``self.design_matrix``. This method also sets the values of ``self.data`` and
        ``self.env``.

        It also populates the dictionary ``self.terms_info`` with information related to each term,
        such as the kind, the columns they occupy in the design matrix and the names of the columns.

        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.
        """
        self.data = data
        self.env = env
        self.design_matrix = np.column_stack([term.data for term in self.terms])
        start = 0
        for term in self.terms:
            if term.data.ndim == 2:
                delta = term.data.shape[1]
            else:
                delta = 1
            self.slices[term.name] = slice(start, start + delta)
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
        if not self.evaluated:
            raise ValueError("Can't evaluate new data on unevaluated matrix.")
        new_instance = self.__class__(self.terms)
        new_instance.data = data
        new_instance.env = self.env
        new_instance.design_matrix = np.column_stack([t.eval_new_data(data) for t in self.terms])
        new_instance.evaluated = True
        return new_instance

    def as_dataframe(self):
        """Returns `self.design_matrix` as a pandas.DataFrame."""
        colnames = [term.get_labels() for term in self.terms]
        data = pd.DataFrame(self.design_matrix, columns=list(flatten_list(colnames)))
        return data

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
        if term not in self.slices:
            raise ValueError(f"'{term}' is not a valid term name")
        return self.design_matrix[:, self.slices[term]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # string = [f"'{k}': {{{spacify(term_str(v))}\n}}" for k, v in self.terms_info.items()]
        # string = multilinify(string)
        string = [
            f"shape: {self.design_matrix.shape}",
            # f"terms: {{{spacify(string)}\n}}",
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
        The data frame where variables are taken from.
    env: Environment
        The environment where values and functions are taken from.

    Attributes
    ----------
    design_matrix: np.array
        A 2 dimensional numpy array with the values of the design matrix.
    evaluated: bool
        Indicates if the terms have been evaluated at least once. The terms must have been evaluated
        before calling ``self._evaluate_new_data()`` because we must know the kind of each term
        to correctly handle the new data passed and the terms here.
    terms_info: dict
        A dictionary that holds information related to each of the group specific terms, such as
        the matrices ``"Xi"`` and ``"Ji"``, ``"cols"``, ``"kind"``, and ``"labels"``. If
        ``"kind"`` is ``"categoric"``, it also contains ``"groups"``, ``"encoding"``, ``"levels"``,
        and ``"reference"``. The keys are given by the term names.
    """

    def __init__(self, terms):
        self.terms = terms
        self.data = None
        self.env = None
        self.design_matrix = np.zeros((0, 0))
        self.slices = {}
        self.evaluated = False

    def _evaluate(self, data, env):
        """Evaluate group specific terms.

        This evaluates ``self.terms`` inside the data mask provided by ``data`` and the environment
        ``env``. It updates ``self.design_matrix`` with the result from the evaluation of each
        term.

        This method also sets the values of ``self.data`` and ``self.env``. It also populates
        the dictionary ``self.terms_info`` with information related to each term,such as the kind,
        the columns and rows they occupy in the design matrix and the names of the columns.

        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.
        """
        self.data = data
        self.env = env
        self.design_matrix = np.column_stack([term.data for term in self.terms])
        start = 0
        for term in self.terms:
            # TODO: I think everything we pass here has two columns...
            if term.data.ndim == 2:
                delta = term.data.shape[1]
            else:
                delta = 1
            self.slices[term.name] = slice(start, start + delta)
            start += delta
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
        new_instance.data = data
        new_instance.env = self.env
        new_instance.design_matrix = np.column_stack([t.eval_new_data(data) for t in self.terms])
        new_instance.evaluated = True
        return new_instance

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
        if term not in self.slices:
            raise ValueError(f"'{term}' is not a valid term name")
        return self.design_matrix[:, self.slices[term]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        # string = [f"'{k}': {{{spacify(term_str(v))}\n}}" for k, v in self.terms_info.items()]
        # string = multilinify(string)
        string = [
            f"shape: {self.design_matrix.shape}",
            # f"terms: {{{spacify(string)}\n}}",
        ]
        return f"GroupEffectsMatrix({wrapify(spacify(multilinify(string)))}\n)"


def design_matrices(formula, data, na_action="drop", env=0):
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
    env: integer
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

    env = Environment.capture(env, reference=1)

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

    design = DesignMatrices(description, data, env)
    return design


# Utils
def term_str(term):
    if term["kind"] == "interaction":
        string_list = [f"{k}: {v}" for k, v in term.items() if k not in ["terms", "Xi", "Ji"]]
        string_vars = [f"'{k}': {{{spacify(term_str(v))}\n}}" for k, v in term["terms"].items()]
        string = multilinify(string_list + [f"vars: {{{spacify(multilinify(string_vars))}\n}}"])
    else:
        string = multilinify([f"{k}: {v}" for k, v in term.items() if k not in ["Xi", "Ji"]])
    return string


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
