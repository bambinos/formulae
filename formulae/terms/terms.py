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


class Intercept:
    """Internal representation of a model intercept."""

    def __init__(self):
        self.name = "Intercept"
        self._type = "Intercept"
        self.data = None
        self.len = None
        self.metadata = {"type": "intercept"}

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return hash(self._type)

    def __add__(self, other):
        """Addition operator.

        Generally this operator is used to explicitly add an intercept to a model. However, there
        may be cases where the result is not a ``Model``, or does not contain an intercept.

        * ``"1 + 0"`` and ``"1 + (-1)"`` return an empty model.
        * ``"1 + 1"`` returns an intercept.
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
        else:
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
            if self in other.common_terms:
                return Model()
            else:
                return self
        else:
            return NotImplemented

    def __or__(self, other):
        """Group-specific operator. Creates group-specific intercept.

        This operation is usually surrounded by parenthesis. It is not actually required. They
        are always used because ``|`` has lower precedence that the other common operators.

        This operator is distributed over the right-hand side, which means ``(1|g + h)`` is
        equivalent to ``(1|g) + (1|h)``.
        """
        if isinstance(other, Term):
            return GroupSpecificTerm(self, other)
        elif isinstance(other, Model):
            products = product([self], other.common_terms)
            terms = [GroupSpecificTerm(p[0], p[1]) for p in products]
            return Model(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}()"

    @property
    def var_names(self):
        """Returns empty set, no variables are used in the intercept."""
        return set()

    def set_type(self, data, eval_env):  # pylint: disable = unused-argument
        """Sets length of the intercept."""
        # Nothing goes here as the type is given by the class.
        # Only works with DataFrames or Series so far
        self.len = data.shape[0]

    def set_data(self, encoding):  # pylint: disable = unused-argument
        """Creates data for the intercept.

        It sets ``self.data`` equal to a numpy array of ones of length ``(self.len, 1)``.
        """
        self.data = np.ones((self.len, 1))

    def eval_new_data(self, data):
        """Returns data for a new intercept.

        The length of the new intercept is given by the number of rows in ``data``.
        """
        # it assumes data is a pandas DataFrame now
        return np.ones((data.shape[0], 1))


class NegatedIntercept:
    """Internal representation of the opposite of a model intercept.

    This object is created whenever we use ``"0"`` or ``"-1"`` in a model formula. It is not
    expected to appear in a final model. It's here to help us make operations using the
    ``Intercept`` and deciding when to keep it and when to drop it.
    """

    def __init__(self):
        self.name = "NegatedIntercept"
        self._type = "Intercept"

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
        else:
            return NotImplemented

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __or__(self, other):
        raise ValueError("At least include an intercept in '|' operation")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}()"

    @property
    def var_names(self):
        # This method should never be called. Leaving a set() to avoid harmless error.
        return set()

    def set_type(self, *args, **kwargs):
        # This method should never be called. Leaving a pass to avoid harmless error.
        pass

    def set_data(self, *args, **kwargs):
        # This method should never be called. Leaving a pass to avoid harmless error.
        pass


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
    data: dict
        The values associated with the term as they go into the design matrix.
    metadata: dict
        Metadata associated with the term. If ``"numeric"`` or ``"categoric"`` it holds additional
        information in the component ``.data`` attribute. If ``"interaction"``, the keys are
        the name of the components and the values are dictionaries holding the metadata.
    _type: string
        Indicates the type of the term. Can be one of ``"numeric"``, ``"categoric"``, or
        ``"interaction"``.
    name: string
        The name of the term as it was originally written in the model formula.
    """

    def __init__(self, *components):
        self.data = None
        self.metadata = {}
        self._type = None
        self.components = []
        self.component_types = None
        for component in components:
            if component not in self.components:
                self.components.append(component)
        self.name = ":".join([str(c.name) for c in self.components])

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
        else:
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
        else:
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
        else:
            return NotImplemented

    def __matmul__(self, other):
        """Simple interaction operator.

        This operator is actually invoked as ``:`` but internally passed as ``@`` because there
        is no ``:`` operator in Python.

        * ``"x : x"`` equals to ``"x"``
        * ``"x : y"`` is the interaction between ``"x"`` and ``"y"``
        * ``x:(y:z)"`` equals to just ``"x:y:z"``
        * ``(x:y):u"`` equals to just ``"x:y:u"``
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
        else:
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
                raise TypeError("Interaction with numeric does not make sense.")
            return Model(self, Term(*self.components, *other.components))
        elif isinstance(other, Model):
            products = product([self], other.common_terms)
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return self + Model(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        """Group-specific operator. Creates group-specific intercept.

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
        else:
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
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "[" + ", ".join([str(component) for component in self.components]) + "]"
        return f"{self.__class__.__name__}({string})"

    def set_type(self, data, eval_env):
        """Set type of the components in the term.

        Calls ``.set_type()`` method on each component in the term. For those components of class
        :class:`.Variable`` it only passes the data mask. For `:class:`.Call` objects it also passes
        the evaluation environment.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from
        eval_env: EvalEnvironment
            The environment where values and functions are taken from.
        """
        # Set the type of the components by calling their set_type method.
        for component in self.components:
            if isinstance(component, Variable):
                component.set_type(data)
            elif isinstance(component, Call):
                component.set_type(data, eval_env)
            else:
                raise ValueError(
                    "Can't set type on Term because at least one of the components "
                    f"is of the unexpected type {type(component)}."
                )
        # Store the type of the components
        self.component_types = {
            component.name: component._type  # pylint: disable = protected-access
            for component in self.components
        }

        # Determine whether this term is numeric, categoric, or an interaction.
        if len(self.components) > 1:
            self._type = "interaction"  # pylint: disable = protected-access
        else:
            self._type = self.components[0]._type  # pylint: disable = protected-access

    def set_data(self, encoding):
        """Obtains and stores the final data object related to this term.

        Calls ``.set_data()`` method on each component in the term. Then, it uses the ``.data``
        attribute on each of them to build ``self.data`` and ``self.metadata``.

        Parameters
        ----------
        encoding: list or dict
            Indicates if it uses full or reduced encoding when the type of the variable is
            categoric.
        """
        if isinstance(encoding, list) and len(encoding) == 1:
            encoding = encoding[0]
        else:
            ValueError("encoding is a list of len > 1")
        for component in self.components:
            encoding_ = False
            if isinstance(encoding, dict):
                encoding_ = encoding.get(component.name, False)
            elif isinstance(encoding, bool):
                encoding_ = encoding
            component.set_data(encoding_)

        if self._type == "interaction":
            self.data = reduce(get_interaction_matrix, [c.data["value"] for c in self.components])
            self.metadata["type"] = "interaction"
            self.metadata["terms"] = {
                c.name: {k: v for k, v in c.data.items() if k != "value"} for c in self.components
            }
        else:
            component = self.components[0]
            self.data = component.data["value"]
            self.metadata = {k: v for k, v in component.data.items() if k != "value"}

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
        if self._type == "interaction":
            result = reduce(
                get_interaction_matrix, [c.eval_new_data(data) for c in self.components]
            )
        else:
            result = self.components[0].eval_new_data(data)
        return result

    @property
    def var_names(self):
        """Returns the name of the variables in the term as a set.

        Loops through each component and updates the set with the ``.var_names`` of each component.

        Returns
        ----------
        var_names: set
            The names of the variables involved in the term.
        """
        var_names = set()
        for component in self.components:
            var_names.update(component.var_names)
        return var_names

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
    factor_type: pandas.core.dtypes.dtypes.CategoricalDtype
        The type assigned to the grouping factor ``factor``. This is useful for when we need to
        create a design matrix for new a new data set.
    """

    def __init__(self, expr, factor):
        self.expr = expr
        self.factor = factor
        self.groups = None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.expr == other.expr and self.factor == other.factor

    def __hash__(self):
        return hash((self.expr, self.factor))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strlist = [
            f"expr= {'  '.join(str(self.expr).splitlines(True))}",
            f"factor= {'  '.join(str(self.factor).splitlines(True))}",
        ]
        return self.__class__.__name__ + "(\n  " + ",\n  ".join(strlist) + "\n)"

    def eval(self, data, eval_env, encoding):
        """Evaluates term.

        First, it evaluates the variable in ``self.factor``, creates an oredered categorical data
        type using its levels, and stores it in ``self.factor_type``. Then, it obtains the
        design matrix for ``self.expr`` to finally produce the matrix for the group specific
        effect.

        The output contains the following information

        * ``"type"``: The type of the ``expr`` term.
        * ``"Xi"``: The design matrix for the ``expr`` term.
        * ``"Ji"``: The design matrix for the ``factor`` term.
        * ``"Zi"``: The design matrix for the group specific term.
        * ``"groups"``: The groups present in ``factor``.

        If ``"type"`` is ``"categoric"``, the output dictionary also contains

        * ``"levels"``: Levels of the term in ``expr``.
        * ``"reference"``: The level taken as baseline.
        * ``"encoding"``: The encoding of the term, either ``"full"`` or ``"reduced"``

        If ``"type"`` is ``"interaction"``, the output dictionary also contains

        * ``"terms"``: Metadata for each of the components in the interaction in ``expr``.

        Parameters
        ----------
        data: pandas.DataFrame
            The data frame where variables are taken from.
        eval_env: EvalEnvironment
            The environment where values and functions are taken from.
        encoding: bool
            Whether to use full or reduced rank encoding when ``expr`` is categoric.

        Returns
        -------
        out: dict
            See above.
        """
        # Factor must be considered categorical, and with full encoding. We set type and obtain
        # data for the factor term manually.

        # Set type on each component to check data is behaved as expected and then
        # manually set type of the components to categoric.
        for comp in self.factor.components:
            if isinstance(comp, Variable):
                comp.set_type(data)
            elif isinstance(comp, Call):
                comp.set_type(data, eval_env)
            else:
                raise ValueError(
                    "Can't set type on Term because at least one of the components "
                    f"is of the unexpected type {type(comp)}."
                )
            comp._type = "categoric"  # pylint: disable = protected-access

        # Store the type of the components.
        # We know they are categoric.
        self.factor.component_types = {comp.name: "categoric" for comp in self.factor.components}

        if len(self.factor.components) > 1:
            self.factor._type = "interaction"  # pylint: disable = protected-access
        else:
            self.factor._type = "categoric"  # pylint: disable = protected-access

        # Pass encoding=True when setting data.
        self.factor.set_data(True)

        # Obtain group names
        groups = []
        for comp in self.factor.components:
            # We're certain they are all categoric with full encoding.
            groups.append([str(lvl) for lvl in comp.data["levels"]])
        self.groups = [":".join(s) for s in list(itertools.product(*groups))]

        self.expr.set_type(data, eval_env)
        self.expr.set_data(encoding)
        Xi = self.expr.data
        Ji = self.factor.data
        Zi = linalg.khatri_rao(Ji.T, Xi.T).T
        out = {
            "type": self.expr.metadata["type"],
            "Xi": Xi,
            "Ji": Ji,
            "Zi": Zi,
            "groups": self.groups,
        }
        if self.expr._type == "categoric":  # pylint: disable = protected-access
            out["levels"] = self.expr.metadata["levels"]
            out["reference"] = self.expr.metadata["reference"]
            out["encoding"] = self.expr.metadata["encoding"]
        elif self.expr._type == "interaction":  # pylint: disable = protected-access
            out["terms"] = self.expr.metadata["terms"]
        return out

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
        out: dict
            Same rules as in :meth:`eval <GroupSpecificTerm.eval>`.
        """

        Xi = self.expr.eval_new_data(data)
        Ji = self.factor.eval_new_data(data)
        Zi = linalg.khatri_rao(Ji.T, Xi.T).T
        out = {
            "type": self.expr.metadata["type"],
            "Xi": Xi,
            "Ji": Ji,
            "Zi": Zi,
            "groups": self.groups,
        }
        if self.expr._type == "categoric":  # pylint: disable = protected-access
            out["levels"] = self.expr.metadata["levels"]
            out["reference"] = self.expr.metadata["reference"]
            out["encoding"] = self.expr.metadata["encoding"]
        elif self.expr._type == "interaction":  # pylint: disable = protected-access
            out["terms"] = self.expr.metadata["terms"]
        return out

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

    def get_name(self):
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
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}({self.term})"

    @property
    def var_names(self):
        """Returns the name of the variables in the response as a set."""
        return self.term.var_names

    def set_type(self, data, eval_env):
        """Set type of the response term."""
        self.term.set_type(data, eval_env)

    def set_data(self, encoding=False):
        """Set data of the response term."""
        self.term.set_data(encoding)


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
        else:
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
        else:
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
        else:
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
        else:
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
        else:
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
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
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
        # Note: Check whether this method is really necessary.
        return [
            comp for term in self.common_terms if isinstance(term, Term) for comp in term.components
        ]

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

    def set_types(self, data, eval_env):
        """Set the type of the common terms in the model.

        Calls ``.set_type()`` method on term in the model.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from
        eval_env: EvalEnvironment
            The environment where values and functions are taken from.
        """
        for term in self.common_terms:
            term.set_type(data, eval_env)

    def _encoding_groups(self):
        components = {}
        for term in self.common_terms:
            if term._type == "interaction":  # pylint: disable = protected-access
                components[term.name] = {
                    c.name: c._type for c in term.components  # pylint: disable = protected-access
                }
            else:
                components[term.name] = term._type  # pylint: disable = protected-access
        # First, group with only categoric terms
        categoric_group = dict()
        for k, v in components.items():
            if v == "categoric":
                categoric_group[k] = [k]
            elif v == "Intercept":
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
                        numeric_groups.append(dict())
                    idx = numeric_group_sets.index(numeric_set)
                    # Prevent full encoding when numeric part is present outside
                    # this numeric-categoric interaction
                    if numeric_part in components.keys():
                        numeric_groups[idx][numeric_part] = []
                    numeric_groups[idx][k] = categoric

        return [categoric_group] + numeric_groups

    def _encoding_bools(self):
        """Determine encodings for terms containing at least one categorical variable.

        This method returns dictionaries with ``True``/``False`` values.
        ``True`` means the categorical variable uses 'levels' dummies.
        ``False`` means the categorial variable uses 'levels - 1' dummies.
        """
        groups = self._encoding_groups()
        l = [pick_contrasts(group) for group in groups]
        result = dict()
        for d in l:
            result.update(d)
        return result

    def eval(self, data, eval_env):
        """Evaluates terms in the model.

        Only common effects are evaluated here. Group specific terms are evaluated individually
        in :class:`GroupEffectsMatrix <formulae.matrices.GroupEffectsMatrix>`.

        Parameters
        ----------
        data: pd.DataFrame
            The data frame where variables are taken from
        eval_env: EvalEnvironment
            The environment where values and functions are taken from.

        Returns
        -------
        result: dict
            A dictionary where keys are the name of the terms and the values are their ``.data``
            attribute.
        """
        self.set_types(data, eval_env)
        encodings = self._encoding_bools()
        result = dict()

        # First, we have to add terms if the encoding implies so.

        # Group specific effects aren't evaluated here -- this may change
        common_terms = self.common_terms.copy()
        for term in common_terms:
            term_encoding = False

            if term.name in encodings.keys():
                term_encoding = encodings[term.name]
            if hasattr(term_encoding, "__len__") and len(term_encoding) > 1:
                # we're in an interaction that added terms.
                # we need to create and evaluate these extra terms.
                # i.e. "y ~ g1:g2", both g1 and g2 categoric, is equivalent to "y ~ g2 + g1:g2"
                # Possibly an interaction adds LOWER order terms, but NEVER HIGHER order terms.
                for (idx, encoding) in enumerate(term_encoding):
                    # Last term never adds any new term, it corresponds to the outer `term`.
                    if idx == len(term_encoding) - 1:
                        term.set_type(data, eval_env)
                        term.set_data(encoding)
                        result[term.name] = term.data
                    else:
                        extra_term = _create_and_eval_extra_term(term, encoding, data, eval_env)
                        result[extra_term.name] = extra_term.data
                        # Finally, add term to self.common_terms object, right before the term
                        # that causes its addition.
                        self.common_terms.insert(self.common_terms.index(term), extra_term)
            else:
                # This term does not add any lower order term, so we just evaluate it as it is.
                term.set_type(data, eval_env)
                term.set_data(term_encoding)
                result[term.name] = term.data
        return result


def _create_and_eval_extra_term(term, encoding, data, eval_env):
    if len(encoding) == 1:
        component_name = list(encoding.keys())[0]
        encoding_ = list(encoding.values())[0]
        component = term.get_component(component_name)
        extra_term = Term(component)
    else:
        component_names = [c.name for c in term.components]
        encoding_ = encoding
        components = [
            term.get_component(name) for name in component_names if name in encoding.keys()
        ]
        extra_term = Term(*components)
    extra_term.set_type(data, eval_env)
    extra_term.set_data(encoding_)
    return extra_term
