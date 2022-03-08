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
            self.response.evaluate(data, env)

        if self.model.common_terms:
            self.common = CommonEffectsMatrix(self.model.common_terms)
            self.common.evaluate(data, env)

        if self.model.group_terms:
            self.group = GroupEffectsMatrix(self.model.group_terms)
            self.group.evaluate(data, env)

    def __getitem__(self, index):
        return (self.response, self.common, self.group)[index]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        entries = []
        if self.response:
            entries += [glue_and_align("Response: ", self.response.design_vector.shape, 30)]

        if self.common:
            entries += [glue_and_align("Common: ", self.common.design_matrix.shape, 30)]

        if self.group:
            entries += [glue_and_align("Group-specific: ", self.group.design_matrix.shape, 30)]

        msg = (
            "DesignMatrices\n\n"
            + glue_and_align("", "(rows, cols)", 30)
            + "\n"
            + "\n".join(entries)
            + "\n\n"
            + "Use .reponse, .common, or .group to access the different members."
        )
        return msg


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
    """

    def __init__(self, term):
        self.term = term
        self.name = self.term.term.name
        self.binary = None  # Not None for categorical variables (either True or False)
        self.data = None
        self.design_vector = None
        self.env = None
        self.kind = None
        self.levels = None  # Not None for categorical variables
        self.success = None  # Not None for binary categorical variables

    def evaluate(self, data, env):
        """Evaluates ``self.term`` inside the data mask provided by ``data`` and
        updates ``self.design_vector`` and ``self.name``.
        """
        self.data = data
        self.env = env
        self.term.set_type(self.data, self.env)
        self.term.set_data()
        self.kind = self.term.term.kind
        self.design_vector = self.term.term.data

        if self.kind == "categoric":
            # NOTE: Why we have self.design_vector.ndim == 1?
            #       Terms are flagged as binary only when built through response[level].
            #       Does it make sense???
            self.binary = self.design_vector.ndim == 1 and len(np.unique(self.design_vector)) == 2
            self.levels = self.term.term.levels
            if self.binary:
                self.success = self.term.term.components[0].reference

    def evaluate_new_data(self, data):
        if self.kind == "proportion":
            return self.term.term.eval_new_data(data)
        raise ValueError("Can't evaluate response term with kind different to 'proportion'")

    def as_dataframe(self):
        """Returns ``self.design_vector`` as a pandas.DataFrame."""
        data = pd.DataFrame(self.design_vector, columns=self.term.term.labels)
        return data

    def __array__(self):
        return self.design_vector

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        entries = [
            f"name: {self.name}",
            f"kind: {self.kind}",
            f"length: {len(self.design_vector)}",
        ]
        if self.kind == "categoric":
            entries += [f"binary: {self.binary}"]
            if self.binary:
                entries += [f"success: {self.success}"]
            else:
                entries += [f"levels: {self.levels}"]
        msg = (
            f"ResponseVector{wrapify(spacify(multilinify(entries, '')))}\n\n"
            "To access the actual design vector do 'np.array(this_obj)'"
        )
        return msg


class CommonEffectsMatrix:
    """Representation of the design matrix for the common effects of a model.

    Parameters
    ----------

    terms : list
        A ...
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
        before calling ``self.evaluate_new_data()`` because we must know the kind of each term
        to correctly handle the new data passed and the terms here.
    terms_info: dict
        A dictionary that holds information related to each of the common specific terms, such as
        ``"cols"``, ``"kind"``, and ``"labels"``. If ``"kind"`` is ``"categoric"``, it also
        contains ``"groups"``, ``"encoding"``, ``"levels"``, and ``"reference"``.
        The keys are given by the term names.
    """

    def __init__(self, terms):
        self.terms = {term.name: term for term in terms}
        self.data = None
        self.env = None
        self.design_matrix = None
        self.evaluated = False
        self.slices = {}

    def evaluate(self, data, env):
        """Obtain design matrix for common effects.

        Evaluates ``self.model`` inside the data mask provided by ``data`` and updates
        ``self.design_matrix``. This method also sets the values of ``self.data`` and
        ``self.env``.

        It also populates the dictionary ``self.slices`` ...

        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.
        """
        self.data = data
        self.env = env
        self.design_matrix = np.column_stack([term.data for term in self.terms.values()])
        start = 0
        for term in self.terms.values():
            if term.data.ndim == 2:
                delta = term.data.shape[1]
            else:
                delta = 1
            self.slices[term.name] = slice(start, start + delta)
            start += delta
        self.evaluated = True

    def evaluate_new_data(self, data):
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
        new_instance = self.__class__(self.terms.values())
        new_instance.data = data
        new_instance.env = self.env
        new_instance.design_matrix = np.column_stack(
            [t.eval_new_data(data) for t in self.terms.values()]
        )
        new_instance.slices = self.slices
        new_instance.evaluated = True
        return new_instance

    def as_dataframe(self):
        """Returns `self.design_matrix` as a pandas.DataFrame."""
        colnames = [term.labels for term in self.terms.values()]
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

    def __array__(self):
        return self.design_matrix

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        entries = []
        for name, term in self.terms.items():
            content = [f"kind: {term.kind}"]
            if hasattr(term, "levels") and term.levels is not None:
                content += [f"levels: {term.levels}"]
            content += [slice_to_column(self.slices[name])]
            entries += [f"{name}{wrapify(spacify(multilinify(content, '')))}"]
        msg = (
            f"CommonEffectsMatrix with shape {self.design_matrix.shape}\n"
            f"Terms:{spacify(multilinify(entries, ''))}\n\n"
            "To access the actual design matrix do 'np.array(this_obj)'"
        )
        return msg


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
        before calling ``self.evaluate_new_data()`` because we must know the kind of each term
        to correctly handle the new data passed and the terms here.
    terms_info: dict
        A dictionary that holds information related to each of the group specific terms, such as
        the matrices ``"Xi"`` and ``"Ji"``, ``"cols"``, ``"kind"``, and ``"labels"``. If
        ``"kind"`` is ``"categoric"``, it also contains ``"groups"``, ``"encoding"``, ``"levels"``,
        and ``"reference"``. The keys are given by the term names.
    """

    def __init__(self, terms):
        self.terms = {term.name: term for term in terms}
        self.data = None
        self.env = None
        self.design_matrix = np.zeros((0, 0))
        self.slices = {}
        self.evaluated = False

    def evaluate(self, data, env):
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
        self.design_matrix = np.column_stack([term.data for term in self.terms.values()])
        start = 0
        for term in self.terms.values():
            # NOTE: I think everything we pass here has two columns...
            if term.data.ndim == 2:
                delta = term.data.shape[1]
            else:
                delta = 1
            self.slices[term.name] = slice(start, start + delta)
            start += delta
        self.evaluated = True

    def evaluate_new_data(self, data):
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

        new_instance = self.__class__(self.terms.values())
        new_instance.data = data
        new_instance.env = self.env
        new_instance.design_matrix = np.column_stack(
            [t.eval_new_data(data) for t in self.terms.values()]
        )
        new_instance.slices = self.slices
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

    def __array__(self):
        return self.design_matrix

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        entries = []
        for name, term in self.terms.items():
            content = [f"kind: {term.kind}", f"groups: {term.groups}"]
            if hasattr(term.expr, "levels") and term.expr.levels is not None:
                content += [f"levels: {term.expr.levels}"]
            content += [slice_to_column(self.slices[name])]
            entries += [f"{name}{wrapify(spacify(multilinify(content, '')))}"]
        msg = (
            f"GroupEffectsMatrix with shape {self.design_matrix.shape}\n"
            f"Terms:{spacify(multilinify(entries, ''))}\n\n"
            "To access the actual design matrix do 'np.array(this_obj)'"
        )
        return msg


def design_matrices(formula, data, na_action="drop", env=0, extra_namespace=None):
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
        all rows with a missing value, ``"error"`` means to raise an error,
        ``"pass"`` means to to keep all. Defaults to ``"drop"``.
    env: integer
        The number of environments we walk up in the stack starting from the function's caller
        to capture the environment where formula is evaluated. Defaults to 0 which means
        the evaluation environment is the environment where ``design_matrices`` is called.
    extra_namespace: dict
        Additional user supplied transformations to include in the environment where the formula
        is evaluated. Defaults to ``None``.

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

    if na_action not in ["drop", "error", "pass"]:
        raise ValueError("'na_action' must be either 'drop', 'error' or 'pass'")

    extra_namespace = extra_namespace or {}

    env = Environment.capture(env, reference=1)
    env = env.with_outer_namespace(extra_namespace)

    description = model_description(formula)

    # Incomplete rows are calculated using columns involved in model formula only
    cols_to_select = description.var_names.intersection(set(data.columns))
    data = data[list(cols_to_select)]

    incomplete_rows = data.isna().any(axis=1)
    incomplete_rows_n = incomplete_rows.sum()

    if incomplete_rows_n > 0:
        if na_action == "pass":
            _log.info(
                "Keeping %s/%s rows with at least one missing value in the dataset.",
                incomplete_rows_n,
                data.shape[0],
            )
        elif na_action == "drop":
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


def slice_to_column(s):
    if s.stop - s.start > 1:
        return f"columns: {s.start}:{s.stop}"
    else:
        return f"column: {s.start}"


def glue_and_align(key, value, width):
    key = str(key)
    value = str(value)
    key_n = len(key)
    value_n = len(value)
    if width > (key_n + value_n):
        return key + value.rjust(width - key_n)
    else:
        return key + value


# Idea: Have a TermList class instead of having to use dictionaries?
