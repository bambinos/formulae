# pylint: disable = too-many-lines
import itertools
import logging

from copy import deepcopy
from functools import reduce
from itertools import combinations, product

import numpy as np
from scipy import linalg

from formulae.utils import get_interaction_matrix
from formulae.contrasts import pick_contrasts

from formulae.terms.call import Call
from formulae.terms.variable import Variable

_log = logging.getLogger("formulae")

# XTODO: Components have 'value' and terms have 'data'... which one should be kept?
class Intercept:
    """Internal representation of a model intercept."""

    def __init__(self):
        self.name = "Intercept"
        self.kind = "intercept"
        self.data = None
        self.len = None

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(self.kind)

    def __add__(self, other):
        """Addition operator.

        Generally this operator is used to explicitly add an intercept to a model. There may be
        cases where the result is not a ``Model``, or does not contain an intercept.

        * ``"1 + 0"`` and ``"1 + (-1)"`` return an empty model.
        * ``"1 + 1"`` returns a single intercept.
        * ``"1 + x"`` and ``"1 + (x|g)"`` returns a model with both the term and the intercept.
        * ``"1 + (x + y)"`` adds an intercept to the model given by ``x`` and ``y``.
        """
        if isinstance(other, NegatedIntercept):
            return Model()
        elif isinstance(other, type(self)):
            return self
        elif isinstance(other, (Term, GroupSpecificTerm)):
            return Model(self, other)
        elif isinstance(other, Model):
            return Model(self) + other
        else:  # pragma: no cover
            return NotImplemented

    def __sub__(self, other):
        """Subtraction operator.

        This operator removes an intercept from a model if the given model has an intercept.

        * ``"1 - 1"`` returns an empty model.
        * ``"1 - 0"`` and ``"1 - (-1)"`` return an intercept.
        * ``"1 - (x + y)"`` returns the model given by ``x`` and ``y`` unchanged.
        * ``"1 - (1 + x + y)"`` returns the model given by ``x`` and ``y``, removing the intercept.
        """
        if isinstance(other, type(self)):
            return Model()
        elif isinstance(other, NegatedIntercept):
            return self
        elif isinstance(other, Model):
            if any(isinstance(term, type(self)) for term in other.common_terms):
                return Model()
            else:
                return self
        else:  # pragma: no cover
            return NotImplemented

    def __or__(self, other):
        """Group-specific interaction-like operator. Creates a group-specific intercept.

        This operation is usually surrounded by parenthesis. It is not actually required. They
        are always used because ``|`` has lower precedence than any of the other operators except
        ``~``.

        This operator is distributed over the right-hand side, which means ``(1|g + h)`` is
        equivalent to ``(1|g) + (1|h)``.
        """
        if isinstance(other, Term):
            return GroupSpecificTerm(self, other)
        elif isinstance(other, Model):
            products = product([self], other.common_terms)
            terms = [GroupSpecificTerm(p[0], p[1]) for p in products]
            return Model(*terms)
        else:  # pragma: no cover
            return NotImplemented

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __str__(self):  # pragma: no cover
        return f"{self.__class__.__name__}()"

    @property
    def var_names(self):
        """Returns empty set, no variables are used in the intercept."""
        return set()

    def set_type(self, data, env):  # pylint: disable = unused-argument
        """Sets length of the intercept."""
        # Nothing goes here as the type is given by the class.
        self.len = data.shape[0]

    def set_data(self, encoding):  # pylint: disable = unused-argument
        """Creates data for the intercept.

        It sets ``self.data`` equal to a numpy array of ones of length ``(self.len, 1)``.
        """
        self.data = np.ones(self.len, dtype=int)

    def eval_new_data(self, data):
        """Returns data for a new intercept.

        The length of the new intercept is given by the number of rows in ``data``.
        """
        return np.ones(data.shape[0], dtype=int)

    @property
    def labels(self):
        return ["Intercept"]


class NegatedIntercept:
    """Internal representation of the opposite of a model intercept.

    This object is created whenever we use ``"0"`` or ``"-1"`` in a model formula. It is not
    expected to appear in a final model. It's here to help us make operations using the
    ``Intercept`` and deciding when to keep it and when to drop it.
    """

    def __init__(self):
        self.name = "NegatedIntercept"
        self.kind = "intercept"

    def __add__(self, other):
        """Addition operator.

        Generally this operator is used to explicitly remove an from a model.

        * ``"0 + 1"`` returns an empty model.
        * ``"0 + 0"`` returns a negated intercept
        * ``"0 + x"`` returns a model that includes the negated intercept.
        * ``"0 + (x + y)"`` adds an the negated intercept to the model given by ``x`` and ``y``.

        No matter the final result contains the negated intercept, for example if we do something
        like ``"y ~ 0 + x + y + 0"``, the ``Model`` that is obtained removes any negated intercepts
        thay may have been left. They just don't make sense in a model.
        """
        if isinstance(other, type(self)):
            return self
        elif isinstance(other, Intercept):
            return Model()
        elif isinstance(other, (Term, GroupSpecificTerm)):
            return Model(self, other)
        elif isinstance(other, Model):
            return Model(self) + other
        else:  # pragma: no cover
            return NotImplemented

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __or__(self, other):
        raise ValueError("At least include an intercept in '|' operation")

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __str__(self):  # pragma: no cover
        return f"{self.__class__.__name__}()"

    @property
    def var_names(self):  # pragma: no cover
        # This method should never be called. Returning set() to avoid harmless error.
        return set()

    def set_type(self, *args, **kwargs):  # pylint: disable = unused-argument # pragma: no cover
        # This method should never be called. Returning None to avoid harmless error.
        return None

    def set_data(self, *args, **kwargs):  # pylint: disable = unused-argument # pragma: no cover
        # This method should never be called. Returning None to avoid harmless error.
        return None


class Term:
    """Representation of a model term.

    Terms are made of one or more components. Components are instances of :class:`.Variable` or
    :class:`.Call`. Terms with only one component are known as main effects and terms with more than
    one component are known as interaction effects. The order of the interaction is given by the
    number of components in the term.

    Parameters
    ----------
    components: :class:`.Variable` or :class:`.Call`
        Atomic components of a term.

    Attributes
    ----------
    data: np.ndarray
        The values associated with the term as they go into the design matrix.
    kind: string
        Indicates the type of the term.
        Can be one of ``"numeric"``, ``"categoric"``, or ``"interaction"``.
    name: string
        The name of the term as it was originally written in the model formula.
    """

    def __init__(self, *components):
        self.components = []
        for component in components:
            if component not in self.components:
                self.components.append(component)
        self.data = None
        self.kind = None
        self.name = ":".join([str(component.name) for component in self.components])

    def __hash__(self):
        return hash(tuple(self.components))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.components == other.components

    def __add__(self, other):
        """Addition operator. Analogous to set union.

        * ``"x + x"`` is equal to just ``"x"``
        * ``"x + y"`` is equal to a model with both ``x`` and ``y``.
        * ``"x + (y + z)"`` adds ``x`` to model already containing ``y`` and ``z``.
        """
        # x + x -> x
        # x + y -> x + y
        # x:y + u -> x:y + u
        # x:y + u:v -> x:y + u:v
        # x:y + (u + v) -> x:y + u + v
        # f(x) + y -> f(x) + y
        # f(x) + (y + z) -> f(x) + y + z
        if self == other:
            return self
        elif isinstance(other, type(self)):
            return Model(self, other)
        elif isinstance(other, Model):
            return Model(self) + other
        else:  # pragma: no cover
            return NotImplemented

    def __sub__(self, other):
        """Subtraction operator. Analogous to set difference.

        * ``"x - x"`` returns empty model.
        * ``"x - y"`` returns the term ``"x"``.
        * ``"x - (y + z)"`` returns the term ``"x"``.
        """
        # x:y - u -> x:y
        # x:y - u:v -> x:y
        # x:y - (u + v) -> x:y
        # f(x) - y -> f(x)
        # f(x) - (y + z) -> f(x)
        if isinstance(other, type(self)):
            if self.components == other.components:
                return Model()
            else:
                return self
        elif isinstance(other, Model):
            if self in other.terms:
                return Model()
            else:
                return self
        else:  # pragma: no cover
            return NotImplemented

    def __mul__(self, other):
        """Full interaction operator.

        This operator includes both the interaction as well as the main effects involved in the
        interaction. It is a shortcut for ``x + y + x:y``.

        * ``"x * x"`` equals to ``"x"``
        * ``"x * y"`` equals to``"x + y + x:y"``
        * ``"x:y * u"`` equals to ``"x:y + u + x:y:u"``
        * ``"x:y * u:v"`` equals to ``"x:y + u:v + x:y:u:v"``
        * ``"x:y * (u + v)"`` equals to ``"x:y + u + v + x:y:u + x:y:v"``
        """
        if self == other:
            return self
        elif isinstance(other, type(self)):
            if len(other.components) == 1 and isinstance(other.components[0].name, (int, float)):
                raise TypeError("Interaction with numeric does not make sense.")
            return Model(self, other, Term(*deepcopy(self.components), *deepcopy(other.components)))
        elif isinstance(other, Model):
            products = product([self], other.common_terms)
            terms = [self] + other.common_terms
            iterms = [
                Term(*deepcopy(p[0].components), *deepcopy(p[1].components)) for p in products
            ]
            return Model(*terms) + Model(*iterms)
        else:  # pragma: no cover
            return NotImplemented

    def __matmul__(self, other):
        """Simple interaction operator.

        This operator is actually invoked as ``:`` but internally passed as ``@`` because there
        is no ``:`` operator in Python.

        * ``"x : x"`` equals to ``"x"``
        * ``"x : y"`` is the interaction between ``"x"`` and ``"y"``
        * ``x:(y:z)"`` equals to ``"x:y:z"``
        * ``(x:y):u"`` equals to ``"x:y:u"``
        * ``"(x:y):(u + v)"`` equals to ``"x:y:u + x:y:v"``
        """
        if self == other:
            return self
        elif isinstance(other, type(self)):
            if len(other.components) == 1 and isinstance(other.components[0].name, (int, float)):
                raise TypeError("Interaction with numeric does not make sense.")
            return Term(*self.components, *other.components)
        elif isinstance(other, Model):
            products = product([self], other.common_terms)
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return Model(*iterms)
        else:  # pragma: no cover
            return NotImplemented

    def __truediv__(self, other):
        """Division interaction operator.

        * ``"x / x"`` equals to just ``"x"``
        * ``"x / y"`` equals to ``"x + x:y"``
        * ``"x / z:y"`` equals to ``"x + x:z:y"``
        * ``"x / (z + y)"`` equals to ``"x + x:z + x:y"``
        * ``"x:y / u:v"`` equals to ``"x:y + x:y:u:v"``
        * ``"x:y / (u + v)"`` equals to ``"x:y + x:y:u + x:y:v"``
        """
        if self == other:
            return self
        elif isinstance(other, type(self)):
            if len(other.components) == 1 and isinstance(other.components[0].name, (int, float)):
                raise TypeError("Interaction with numbers does not make sense.")
            return Model(self, Term(*self.components, *other.components))
        elif isinstance(other, Model):
            products = product([self], other.common_terms)
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return self + Model(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        """Group-specific operator. Creates a group-specific term.

        Intercepts are implicitly added.

        * ``"x|g"`` equals to ``"(1|g) + (x|g)"``

        Distributive over right hand side

        * ``"(x|g + h)"`` equals to ``"(1|g) + (1|h) + (x|g) + (x|h)"``
        """
        if isinstance(other, Term):
            # Only accepts terms, call terms and interactions.
            # Adds implicit intercept.
            terms = [GroupSpecificTerm(Intercept(), other), GroupSpecificTerm(self, other)]
            return Model(*terms)
        elif isinstance(other, Model):
            intercepts = [
                GroupSpecificTerm(Intercept(), p[1]) for p in product([self], other.common_terms)
            ]
            slopes = [GroupSpecificTerm(p[0], p[1]) for p in product([self], other.common_terms)]
            return Model(*intercepts, *slopes)
        else:  # pragma: no cover
            return NotImplemented

    def __pow__(self, other):
        """Power operator.

        It leaves the term as it is. For a power in the math sense do ``I(x ** n)`` or ``{x ** n}``.
        """
        c = other.components
        if len(c) == 1 and isinstance(c[0].name, int) and c[0].name >= 1:
            _log.warning(
                "Exponentiation on an individual variable returns the variable as it is.\n"
                "Use {%s**%s} or I(%s**%s) to compute the math power.",
                self.name,
                c[0].name,
                self.name,
                c[0].name,
            )
            return self
        else:  # pragma: no cover
            return NotImplemented

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __str__(self):  # pragma: no cover
        string = "[" + ", ".join([str(component) for component in self.components]) + "]"
        return f"{self.__class__.__name__}({string})"

    def set_type(self, data, env):
        """Set type of the components in the term.

        Calls ``.set_type()`` method on each component in the term. For those components of class
        :class:`.Variable`` it only passes the data mask. For `:class:`.Call` objects it also passes
        the evaluation environment.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.
        """
        # Set the type of the components by calling their set_type method.
        for component in self.components:
            if isinstance(component, Variable):
                component.set_type(data)
            elif isinstance(component, Call):
                component.set_type(data, env)
            else:
                raise ValueError(
                    "Can't set type on Term because at least one of the components "
                    f"is of the unexpected type {type(component)}."
                )

        # Determine whether this term is numeric, categoric, or an interaction.
        if len(self.components) > 1:
            self.kind = "interaction"
        else:
            self.kind = self.components[0].kind

    def set_data(self, spans_intercept):
        """Obtains and stores the final data object related to this term.

        Calls ``.set_data()`` method on each component in the term. Then, it uses the ``.data``
        attribute on each of them to build ``self.data`` and ``self.metadata``.

        Parameters
        ----------
        encoding: dict or bool
            Indicates if it uses full or reduced encoding when the type of the variable is
            categoric.
        """

        for component in self.components:
            spans_intercept_ = False
            if isinstance(spans_intercept, dict):
                spans_intercept_ = spans_intercept.get(component.name, False)
            elif isinstance(spans_intercept, bool):
                spans_intercept_ = spans_intercept
            else:
                raise ValueError(f"Encoding is of unexpected type {type(spans_intercept_)}.")

            component.set_data(spans_intercept_)

        if self.kind == "interaction":
            self.data = reduce(get_interaction_matrix, [c.value for c in self.components])
        else:
            self.data = self.components[0].value

    def eval_new_data(self, data):
        """Evaluates the term with new data.

        Calls ``.eval_new_data()`` method on each component in the term and combines the results
        appropiately.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from

        Returns
        ----------
        result: np.array
            The values resulting from evaluating this term using the new data.
        """
        if self.kind == "interaction":
            result = reduce(
                get_interaction_matrix, [c.eval_new_data(data) for c in self.components]
            )
        else:
            result = self.components[0].eval_new_data(data)
        return result

    def get_component(self, name):  # pylint: disable = inconsistent-return-statements
        """Returns a component by name.

        Parameters
        ----------
        name: string
            The name of the component to return.

        Returns
        -------
        component: `:class:`.Variable` or `:class:`.Call`
            The component with name ``name``.
        """

        for component in self.components:
            if component.name == name:
                return component

    @property
    def var_names(self):
        """Returns the name of the variables in the term as a set.

        Loops through each component and updates the set with the ``.var_names`` of each component.

        Returns
        ----------
        var_names: set
            The names of the variables involved in the term.
        """
        var_names = set().union(*[component.var_names for component in self.components])
        return var_names

    @property
    def labels(self):
        """Obtain labels of the columns in the design matrix associated with this Term"""
        if self.kind is None:
            labels = None
        elif self.kind == "interaction":
            labels = []
            for component in self.components:
                labels.append(component.labels)
            labels = [":".join(str_tuple) for str_tuple in list(itertools.product(*labels))]
        else:
            labels = self.components[0].labels
        return labels

    @property
    def levels(self):
        """Obtain levels of the columns in the design matrix associated with this Term

        It is like .labels, without the name of the terms
        """
        if self.kind is None or self.kind == "numeric":
            levels = None
        elif self.kind == "interaction":
            levels = []
            for component in self.components:
                if component.contrast_matrix is not None:
                    levels.append(component.contrast_matrix.labels)
            if levels:
                levels = [", ".join(str_tuple) for str_tuple in list(itertools.product(*levels))]
        else:
            component = self.components[0]
            if component.is_response and component.reference is not None:
                levels = None
            else:
                levels = component.contrast_matrix.labels
        return levels

    @property
    def spans_intercept(self):
        """Does this term spans the intercept?

        True if all the components span the intercept
        """
        return all(component.spans_intercept for component in self.components)


class GroupSpecificTerm:
    """Representation of a group specific term.

    Group specific terms are of the form ``(expr | factor)``. The expression ``expr`` is evaluated
    as a model formula with only common effects and produces a model matrix following the rules
    for common terms. ``factor`` is inspired on factors in R, but here it is evaluated as an ordered
    pandas.CategoricalDtype object.

    The operator ``|`` works as in R package lme4. As its authors say: "One way to think about the
    vertical bar operator is as a special kind of interaction between the model matrix and the
    grouping factor. This interaction ensures that the columns of the model matrix have different
    effects for each level of the grouping factor"

    Parameters
    ----------
    expr: :class:`.Intercept` or :class:`.Term`
        The term for which we want to have a group specific term.
    factor: :class:`.Term`
        The factor that determines the groups in the group specific term.

    Attributes
    ----------
    data: np.ndarray
        The values associated with the term as they go into the design matrix.
    metadata: dict
        Metadata associated with the term. If ``"numeric"`` or ``"categoric"`` it holds additional
        information in the component ``.data`` attribute. If ``"interaction"``, the keys are
        the name of the components and the values are dictionaries holding the metadata.
    kind: string
        Indicates the type of the term. Can be one of ``"numeric"``, ``"categoric"``, or
        ``"interaction"``.
    """

    def __init__(self, expr, factor):
        self.expr = expr
        self.factor = factor
        self.data = None
        self.groups = None
        self.kind = None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.expr == other.expr and self.factor == other.factor

    def __hash__(self):
        return hash((self.expr, self.factor))

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __str__(self):  # pragma: no cover
        strlist = [
            f"expr= {'  '.join(str(self.expr).splitlines(True))}",
            f"factor= {'  '.join(str(self.factor).splitlines(True))}",
        ]
        return self.__class__.__name__ + "(\n  " + ",\n  ".join(strlist) + "\n)"

    def set_type(self, data, env):
        # Set type of 'factor'
        # Set type on each component of the factor to check data is behaved as expected and then
        # manually set their type to categoric.
        for component in self.factor.components:
            if isinstance(component, Variable):
                component.set_type(data)
            elif isinstance(component, Call):
                component.set_type(data, env)
            else:
                raise ValueError(
                    "Can't set type on GroupSpecificTerm because at least one of the components "
                    f"is of the unexpected type {type(component)}."
                )
            component.kind = "categoric"

        # Store the type of the components. Factors are considered categorical.
        if len(self.factor.components) > 1:
            self.factor.kind = "interaction"
        else:
            self.factor.kind = "categoric"

        # Set type of 'expr'
        self.expr.set_type(data, env)

    def set_data(self, spans_intercept):
        self.expr.set_data(spans_intercept)
        self.factor.set_data(True)  # Factor is a categorical term that always spans the intercept

        # Obtain group names. These are obtained from the labels of the contrast matrices
        groups = []
        for component in self.factor.components:
            groups.append(component.contrast_matrix.labels)
        self.groups = [":".join(s) for s in list(itertools.product(*groups))]

        Xi, Ji = self.expr.data, self.factor.data
        if Xi.ndim == 1:
            Xi = Xi[:, np.newaxis]
        if Ji.ndim == 1:
            Ji = Ji[:, np.newaxis]

        self.data = linalg.khatri_rao(Ji.T, Xi.T).T  # Zi
        self.kind = self.expr.kind

    def eval_new_data(self, data):
        """Evaluates the term with new data.

        Converts the variable in ``factor`` to the type remembered from the first evaluation and
        produces the design matrix for this grouping, calls ``.eval_new_data()`` on ``self.expr``
        to obtain the design matrix for the ``expr`` side, then computes the design matrix
        corresponding to the group specific effect.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from.

        Returns
        ----------
        Zi: np.ndarray
        """
        Xi = self.expr.eval_new_data(data)
        Ji = self.factor.eval_new_data(data)
        if Xi.ndim == 1:
            Xi = Xi[:, np.newaxis]
        if Ji.ndim == 1:
            Ji = Ji[:, np.newaxis]
        Zi = linalg.khatri_rao(Ji.T, Xi.T).T
        return Zi

    @property
    def var_names(self):
        """Returns the name of the variables in the term as a set.

        Obtains both the variables in the ``expr`` as well as the variables in ``factor``.

        Returns
        ----------
        var_names: set
            The names of the variables involved in the term.
        """
        expr_names = self.expr.var_names.copy()
        factor_names = self.factor.var_names.copy()
        return expr_names.union(factor_names)

    @property
    def name(self):
        """Obtain string representation of the name of the term.

        Returns
        ----------
        name: str
            The name of the term, such as ``1|g`` or ``var|g``.
        """
        name = ""
        if isinstance(self.expr, Intercept):
            name += "1|"
        elif isinstance(self.expr, Term):
            name += f"{self.expr.name}|"
        else:
            raise ValueError("Invalid LHS expression for group specific term")

        if isinstance(self.factor, Term):
            name += self.factor.name
        else:
            raise ValueError("Invalid RHS expression for group specific term")
        return name

    @property
    def labels(self):
        if self.kind is None:
            labels = None
        if self.kind == "intercept":
            levels = ["1"]
        else:
            levels = self.expr.labels
        labels = [f"{level}|{group}" for group in self.factor.labels for level in levels]
        return labels


class Response:
    """Representation of a response term.

    It is mostly a wrapper around :class:`.Term`.

    Parameters
    ----------
    term: :class:`.Term`
        The term we want to take as response in the model. Must contain only one component.

    """

    def __init__(self, term):
        if isinstance(term, Term):
            n = len(term.components)
            if n == 1:
                self.term = term
                self.term.components[0].is_response = True
            else:
                raise ValueError(f"The response term must contain only one component, not {n}.")
        else:
            raise ValueError(f"The response term must be of class Term, not {type(term)}.")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.term == other.term

    def __add__(self, other):
        """Modelled as operator.

        The operator is ``~``, but since it is not an operator in Python, we internally replace it
        with ``+``. It means the LHS is taken as the response, and the RHS as the predictor.
        """
        if isinstance(other, (Term, GroupSpecificTerm, Intercept)):
            return Model(other, response=self)
        elif isinstance(other, Model):
            return other.add_response(self)
        else:  # pragma: no cover
            return NotImplemented

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __str__(self):  # pragma: no cover
        return f"{self.__class__.__name__}({self.term})"

    @property
    def var_names(self):
        """Returns the name of the variables in the response as a set."""
        return self.term.var_names

    def set_type(self, data, env):
        """Set type of the response term."""
        self.term.set_type(data, env)

    def set_data(self):
        """Set data of the response term."""
        self.term.set_data(spans_intercept=True)


ACCEPTED_TERMS = (Term, GroupSpecificTerm, Intercept, NegatedIntercept)


class Model:
    """Representation of a model.

    Parameters
    ----------
    terms: :class:`.Term`
        This object can be instantiated with one or many terms.
    response::class:`.Response`
        The response term. Defaults to ``None`` which means there is no response.
    """

    def __init__(self, *terms, response=None):
        if isinstance(response, Response) or response is None:
            self.response = response
        else:
            raise ValueError("Response must be of class Response.")
        if all(isinstance(term, ACCEPTED_TERMS) for term in terms):
            self.common_terms = [term for term in terms if not isinstance(term, GroupSpecificTerm)]
            self.group_terms = [term for term in terms if isinstance(term, GroupSpecificTerm)]
        else:
            raise ValueError("There is a least one term of an unexpected class.")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        equal_terms = set(self.terms) == set(other.terms)
        equal_response = self.response == other.response
        return equal_terms and equal_response

    def __add__(self, other):
        """Addition operator. Analogous to set union.

        Adds terms to the model and returns the model.

        Returns
        -------
        self: :class:`.Model`
            The same model object with the added term(s).
        """
        if isinstance(other, NegatedIntercept):
            return self - Intercept()
        elif isinstance(other, (Term, GroupSpecificTerm, Intercept)):
            return self.add_term(other)
        elif isinstance(other, type(self)):
            for term in other.terms:
                self.add_term(term)
            return self
        else:  # pragma: no cover
            return NotImplemented

    def __sub__(self, other):
        """Subtraction operator. Analogous to set difference.

        * ``"(x + y) - (x + u)"`` equals to ``"y + u"``..
        * ``"(x + y) - x"`` equals to ``"y"``.
        * ``"(x + y + (1 | g)) - (1 | g)"`` equals to ``"x + y"``.

        Returns
        -------
        self: :class:`.Model`
            The same model object with the removed term(s).
        """
        if isinstance(other, type(self)):
            for term in other.terms:
                if term in self.common_terms:
                    self.common_terms.remove(term)
                if term in self.group_terms:
                    self.group_terms.remove(term)
            return self
        elif isinstance(other, (Term, Intercept)):
            if other in self.common_terms:
                self.common_terms.remove(other)
            return self
        elif isinstance(other, GroupSpecificTerm):
            if other in self.group_terms:
                self.group_terms.remove(other)
            return self
        else:  # pragma: no cover
            return NotImplemented

    def __matmul__(self, other):
        """Simple interaction operator.

        * ``"(x + y) : (u + v)"`` equals to ``"x:u + x:v + y:u + y:v"``.
        * ``"(x + y) : u"`` equals to ``"x:u + y:u"``.
        * ``"(x + y) : f(u)"`` equals to ``"x:f(u) + y:f(u)"``.

        Returns
        -------
        model: :class:`.Model`
            A new instance of the model with all the interaction terms computed.
        """
        if isinstance(other, type(self)):
            products = product(self.common_terms, other.common_terms)
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return Model(*iterms)
        elif isinstance(other, Term):
            products = product(self.common_terms, [other])
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return Model(*iterms)
        else:  # pragma: no cover
            return NotImplemented

    def __mul__(self, other):
        """Full interaction operator.

        * ``"(x + y) * (u + v)"`` equals to ``"x + y + u + v + x:u + x:v + y:u + y:v"``.
        * ``"(x + y) * u"`` equals to ``"x + y + u + x:u + y:u"``.

        Returns
        -------
        model: :class:`.Model`
            A new instance of the model with all the interaction terms computed.
        """
        if self == other:
            return self
        elif isinstance(other, type(self)):
            if len(other.common_terms) == 1:
                components = other.common_terms[0].components
                if len(components) == 1 and isinstance(components, (int, float)):
                    raise TypeError("Interaction with numeric does not make sense.")
            products = product(self.common_terms, other.common_terms)
            terms = self.common_terms + other.common_terms
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return Model(*terms) + Model(*iterms)
        elif isinstance(other, Term):
            if len(other.components) == 1 and isinstance(other.components[0].name, (int, float)):
                raise TypeError("Interaction with numeric does not make sense.")
            products = product(self.common_terms, [other])
            terms = self.common_terms + [other]
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return Model(*terms) + Model(*iterms)
        else:  # pragma: no cover
            return NotImplemented

    def __pow__(self, other):
        """Power of a set made of :class:`.Term`

        Computes all interactions up to order ``n`` between the terms in the set.

        * ``"(x + y + z) ** 2"`` equals to ``"x + y + z + x:y + x:z + y:z"``.

        Returns
        -------
        model: :class:`.Model`
            A new instance of the model with all the terms computed.
        """
        if isinstance(other, Term) and len(other.components) == 1:
            value = other.components[0].name
            if isinstance(value, int) and value >= 1:
                comb = [
                    list(p) for i in range(2, value + 1) for p in combinations(self.common_terms, i)
                ]
            iterms = [Term(*[comp for term in terms for comp in term.components]) for terms in comb]
            return self + Model(*iterms)
        else:
            raise ValueError("Power must be a positive integer.")

    def __truediv__(self, other):
        """Division interaction operator.

        * ``"(x + y) / z"`` equals to ``"x + y + x:y:z"``.
        * ``"(x + y) / (u + v)"`` equals to ``"x + y + x:y:u + x:y:v"``.

        Returns
        -------
        model: :class:`.Model`
            A new instance of the model with all the terms computed.
        """
        if isinstance(other, Term):
            return self.add_term(Term(*self.common_components + other.components))
        elif isinstance(other, Model):
            iterms = [Term(*self.common_components, comp) for comp in other.common_components]
            return self + Model(*iterms)
        else:  # pragma: no cover
            return NotImplemented

    def __or__(self, other):
        """Group specific term operator.

        Only _models_ ``"0 + x"`` arrive here.

        * ``"(0 + x | g)"`` equals to ``"(x|g)"``.
        * ``"(0 + x | g + y)"`` equals to ``"(x|g) + (x|y)"``.

        There are several edge cases to handle here. See in-line comments.

        Returns
        -------
        model: :class:`.Model`
            A new instance of the model with all the terms computed.
        """

        # If only one term in the expr, resolve according to the type of the term.
        if len(self.common_terms) == 1:
            return self.common_terms[0] | other

        # Handle intercept
        if Intercept() in self.common_terms and NegatedIntercept() in self.common_terms:
            # Explicit addition and negation -> remove both -> no intercept
            self.common_terms.remove(Intercept())
            self.common_terms.remove(NegatedIntercept())
        elif NegatedIntercept() in self.common_terms:
            # Negation -> remove negation and do not add intercept
            self.common_terms.remove(NegatedIntercept())
        elif Intercept() not in self.common_terms:
            # No negation and no explicit intercept -> implicit intercept
            self.common_terms.insert(0, Intercept())
        if isinstance(other, Term):
            products = product(self.common_terms, [other])
            terms = [GroupSpecificTerm(p[0], p[1]) for p in products]
            return Model(*terms)
        elif isinstance(other, type(self)):
            products = product(self.common_terms, other.common_terms)
            terms = [GroupSpecificTerm(p[0], p[1]) for p in products]
            return Model(*terms)
        else:  # pragma: no cover
            return NotImplemented

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __str__(self):  # pragma: no cover
        terms = [str(term) for term in self.common_terms]
        if self.response is not None:
            terms.insert(0, str(self.response))
        string = ",\n  ".join([str(term) for term in terms])

        if self.group_terms:
            group_terms = ",\n".join([str(term) for term in self.group_terms])
            if len(string) > 0:
                string += ",\n  "
            string += "  ".join(group_terms.splitlines(True))

        return f"{self.__class__.__name__}(\n  {string}\n)"

    def add_response(self, term):
        """Add response term to model description.

        This method is called when something like ``"y ~ x + z"`` appears in a model formula.

        This method is called via special methods such as :meth:`Response.__add__`.

        Returns
        -------
        self: :class:`.Model`
            The same model object but now with a reponse term.
        """
        if isinstance(term, Response):
            self.response = term
            return self
        else:
            raise ValueError("not Response")

    def add_term(self, term):
        """Add term to model description.

        The term added can be of class :class:`.Intercept` :class:`.Term`, or
        :class:`.GroupSpecificTerm`. It appends the new term object to the list of common terms or
        group specific terms as appropriate.

        This method is called via special methods such as :meth:`__add__`.

        Returns
        -------
        self: :class:`.Model`
            The same model object but now containing the new term.
        """
        if isinstance(term, GroupSpecificTerm):
            if term not in self.group_terms:
                self.group_terms.append(term)
            return self
        elif isinstance(term, (Term, Intercept)):
            if term not in self.common_terms:
                self.common_terms.append(term)
            return self
        else:
            raise ValueError(f"Can't add an object of class {type(term)} to Model.")

    @property
    def terms(self):
        """Terms in the model.

        Returns
        -------
        terms: list
            A list containing both common and group specific terms.
        """
        return self.common_terms + self.group_terms

    @property
    def common_components(self):
        """Components in common terms in the model.

        Returns
        -------
        components: list
            A list containing all components from common terms in the model.
        """
        return [c for term in self.common_terms if isinstance(term, Term) for c in term.components]

    @property
    def var_names(self):
        """Get the name of the variables in the model.

        Returns
        -------
        var_names: set
            The names of all variables in the model.
        """

        var_names = set()
        for term in self.terms:
            var_names.update(term.var_names)
        if self.response is not None:
            var_names.update(self.response.var_names)
        return var_names

    def set_types(self, data, env):
        """Set the type of the terms in the model.

        Calls ``.set_type()`` method on term in the model.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.
        """
        for term in self.terms:
            term.set_type(data, env)

    def _get_encoding_groups(self):
        components = {}
        for term in self.common_terms:
            if term.kind == "interaction":
                components[term.name] = {c.name: c.kind for c in term.components}
            else:
                components[term.name] = term.kind

        # First, group with only categoric terms
        categoric_group = {}
        for k, v in components.items():
            if v == "categoric":
                categoric_group[k] = [k]
            elif v == "intercept":
                categoric_group[k] = []
            elif isinstance(v, dict):  # interaction
                # If all categoric terms in the interaction
                if all(v_ == "categoric" for v_ in v.values()):
                    categoric_group[k] = list(v.keys())

        # Determine groups of numerics
        numeric_group_sets = []
        numeric_groups = []
        for k, v in components.items():
            # v is dict when interaction, otherwise is string.
            if isinstance(v, dict):
                categoric = [k_ for k_, v_ in v.items() if v_ == "categoric"]
                numeric = [k_ for k_, v_ in v.items() if v_ == "numeric"]
                # if it is an interaction with both categoric and numeric terms
                if categoric and numeric:
                    numeric_set = set(numeric)
                    numeric_part = ":".join(numeric)
                    if numeric_set not in numeric_group_sets:
                        numeric_group_sets.append(numeric_set)
                        numeric_groups.append({})
                    idx = numeric_group_sets.index(numeric_set)
                    # Prevent full encoding when numeric part is present outside
                    # this numeric-categoric interaction
                    if numeric_part in components:
                        numeric_groups[idx][numeric_part] = []
                    numeric_groups[idx][k] = categoric

        return [categoric_group] + numeric_groups

    def _get_encoding_bools(self):
        """Determine encodings for terms containing at least one categorical variable.

        This method returns dictionaries with ``True``/``False`` values.
        ``True`` means the categorical variable spans the intercept.
        ``False`` means the categorial variable does not span the intercept.
        """
        groups = self._get_encoding_groups()
        l = [pick_contrasts(group) for group in groups]
        result = {}
        for d in l:
            result.update(d)
        return result

    def add_extra_terms(self, encodings, data, env):
        # Adds additional terms in the common part in case they're needed for full rankness
        common_terms = self.common_terms.copy()
        for term in common_terms:
            encoding = encodings.get(term.name)
            if hasattr(encoding, "__len__") and len(encoding) > 1:
                # Last encoding is the one for the original term
                for subencoding in encoding[:-1]:
                    extra_term = create_extra_term(term, subencoding, data, env)
                    self.common_terms.insert(self.common_terms.index(term), extra_term)

    def eval(self, data, env):
        """Evaluates terms in the model.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.
        """
        # Set types on all terms
        self.set_types(data, env)

        # Evaluate common terms
        encodings = self._get_encoding_bools()
        self.add_extra_terms(encodings, data, env)

        # Need to get encodings again after creating possible extra terms
        encodings = self._get_encoding_bools()

        for term in self.common_terms:
            if term.name in encodings:
                # Since we added extra terms before, we can assume 'encodings' has lists of length 1
                encoding = encodings[term.name][0]
            else:
                encoding = False
            term.set_data(encoding)

        # Evaluate group-specific terms
        for term in self.group_terms:
            encoding = True
            # If both (1|g) and (x|g) are in the model, then the encoding for x is False.
            if not isinstance(term.expr, Intercept):
                for t in self.group_terms:
                    if t.factor == term.factor and isinstance(t.expr, Intercept):
                        encoding = False
            term.set_data(encoding)


def create_extra_term(term, encoding, data, env):
    """
    If there are numeric components it means this is an interaction term that has both numeric
    and categoric components. The categoric part of the term we create comes in 'encoding', we
    then need to add the numeric ones.

    For example, if we have 'h' and 'j' categoric and 'x' numeric and then we do 'x + h:j:x',
    it expands to 'x + j:x + h:j:x' for full-rankness.
    """
    component_names = [component.name for component in term.components]
    components = [term.get_component(name) for name in component_names if name in encoding.keys()]
    components += [component for component in term.components if component.kind == "numeric"]
    extra_term = Term(*deepcopy(components))
    extra_term.set_type(data, env)
    return extra_term
