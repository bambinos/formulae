import logging

from functools import reduce
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy import linalg, sparse

from formulae.utils import get_interaction_matrix
from formulae.contrasts import pick_contrasts

from formulae.terms.call import Call
from formulae.terms.variable import Variable

_log = logging.getLogger("formulae")


class Intercept:
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
        """
        (1|g) -> (1|g); (1|g:h) -> (1|g:h)
        (1 | g + h) -> (1|g) + (1|h)
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
        return set()

    def set_type(self, data, eval_env):  # pylint: disable = unused-argument
        # Nothing goes here as the type is given by the class.
        # Only works with DataFrames or Series so far
        self.len = data.shape[0]

    def set_data(self, encoding):  # pylint: disable = unused-argument
        self.data = np.ones((self.len, 1))

    def eval_new_data(self, data):
        # it assumes data is a pandas DataFrame now
        return np.ones((data.shape[0], 1))


class NegatedIntercept:
    def __init__(self):
        self.name = "NegatedIntercept"
        self._type = "Intercept"

    def __add__(self, other):
        """
        0 + 0 -> 0
        0 + 1 -> <empty>
        0 + a -> 0 + a
        0 + (a|g) -> 0 + (a|g)
        0 + (a + b) -> 0 + a + b
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
    """Representation of a single term.

    A model term can be an intercept, a term made of a single component, a function call,
    an interaction involving components and/or function calls.
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
        """Sum between of ``Term`` and other classes of terms.

        Analogous to set union.
        x + x -> x
        x + y -> x + y
        x:y + u -> x:y + u
        x:y + u:v -> x:y + u:v
        x:y + (u + v) -> x:y + u + v
        f(x) + y -> f(x) + y
        f(x) + (y + z) -> f(x) + y + z
        """
        if self == other:
            return self
        elif isinstance(other, type(self)):
            return Model(self, other)
        elif isinstance(other, Model):
            return Model(self) + other
        else:
            return NotImplemented

    def __sub__(self, other):
        """Difference between a ``Term`` and other classes of terms.

        Analogous to set difference.
        x - x -> ()
        x - y -> x
        x:y - u -> x:y
        x:y - u:v -> x:y
        x:y - (u + v) -> x:y
        f(x) - y -> f(x)
        f(x) - (y + z) -> f(x)
        """
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
        """Full interaction.

        x * x -> x + x + x:x -> x
        x * y -> x + y + x:y
        x:y * u -> x:y + u + x:y:u
        x:y * u:v -> x:y + u:v + x:y:u:v
        x:y * (u + v) -> x:y + u + v + x:y:u + x:y:v
        f(x) * y -> f(x) + y + f(x):y
        f(x) * (y + z) -> f(x) + y + z + f(x):y + f(x):z
        """
        if self == other:
            return self
        elif isinstance(other, type(self)):
            if len(other.components) == 1 and isinstance(other.components[0].name, (int, float)):
                raise TypeError("Interaction with numeric does not make sense.")
            return Model(self, other, Term(*self.components, *other.components))
        elif isinstance(other, Model):
            products = product([self], other.common_terms)
            terms = [self] + other.common_terms
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return Model(*terms) + Model(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        """Simple interaction.

        x:x -> x
        x:y -> x:y
        x:(y:z) -> x:y:z
        (x:y):u -> x:y:u
        (x:y):(u + v) -> x:y:u + x:y:v
        f(x):y -> f(x):y
        f(x):y:z -> f(x):y:z
        f(x):(y:z) -> f(x):y:z
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

    def __pow__(self, other):
        """Power of a ``Term``.

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

    def __truediv__(self, other):
        """Division interaction.

        x / x -> x
        x / y -> x + x:y
        x / z:y -> x + x:z:y
        x / (z + y) -> x + x:z + x:y
        x:y / u:v -> x:y + x:y:u:v
        x:y / (u + v) -> x:y + x:y:u + x:y:v
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
        """Group specific term operation.

        (x|g) -> (1|g) + (x|g)
        (x|g + h) -> (x|g) + (x|h) -> (1|g) + (1|h) + (x|g) + (x|h)
        (x|g:h) -> (1|g:h) + (x|g:h)
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "[" + ", ".join([str(component) for component in self.components]) + "]"
        return f"{self.__class__.__name__}({string})"

    def set_type(self, data, eval_env):
        """Set type of the components in the term.

        Calls ``.set_type()`` method on each component in the term. For those components of class
        ``Variable`` it only passes the data mask. For ``Call`` objects it also passes the
        evaluation environment.
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
        """Obtains and stores the final data object related to this term"""
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
        """Evaluates the term with new data."""
        if self._type == "interaction":
            data = reduce(get_interaction_matrix, [c.eval_new_data(data) for c in self.components])
        else:
            data = self.components[0].eval_new_data(data)
        return data

    @property
    def var_names(self):
        """Returns the name of the variables in the term as a set."""
        var_names = set()
        for component in self.components:
            var_names.update(component.var_names)
        return var_names

    def get_component(self, name):  # pylint: disable = inconsistent-return-statements
        """Returns a component by name

        This method receives a component name and returns the component, either a Variable or Call.
        """

        for component in self.components:
            if component.name == name:
                return component


class GroupSpecificTerm:
    def __init__(self, expr, factor):
        self.expr = expr
        self.factor = factor
        self.factor_type = None

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
        # Note: factor can't be a call or interaction yet.
        if len(self.factor.components) == 1 and isinstance(self.factor.components[0], Variable):
            factor = data[self.factor.name]
            if not hasattr(factor.dtype, "ordered") or not factor.dtype.ordered:
                categories = sorted(factor.unique().tolist())
                type_ = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
                factor = factor.astype(type_)
            else:
                type_ = factor.dtype
            self.factor_type = type_
        else:
            raise ValueError(
                "Factor on right hand side of group specific term must be a single term."
            )

        # Notation as in lme4 paper
        # Note we don't use `drop_first=True` for factor.
        self.expr.set_type(data, eval_env)
        self.expr.set_data(encoding)
        Xi = self.expr.data
        Ji = pd.get_dummies(factor).to_numpy()
        Zi = linalg.khatri_rao(Ji.T, Xi.T).T
        out = {
            "type": self.expr.metadata["type"],
            "Xi": Xi,
            "Ji": Ji,
            "Zi": sparse.coo_matrix(Zi),
            "groups": factor.cat.categories.tolist(),
        }
        if self.expr._type == "categoric":  # pylint: disable = protected-access
            out["levels"] = self.expr.metadata["levels"]
            out["reference"] = self.expr.metadata["reference"]
            out["encoding"] = self.expr.metadata["encoding"]
        elif self.expr._type == "interaction":  # pylint: disable = protected-access
            out["terms"] = self.expr.metadata["terms"]
        return out

    def eval_new_data(self, data):
        """Evaluates the term with new data."""

        # factor uses the same data type that is used in first evaluation.
        factor = data[self.factor.name].astype(self.factor_type)
        Xi = self.expr.eval_new_data(data)
        Ji = pd.get_dummies(factor).to_numpy()
        Zi = linalg.khatri_rao(Ji.T, Xi.T).T
        out = {
            "type": self.expr.metadata["type"],
            "Xi": Xi,
            "Ji": Ji,
            "Zi": sparse.coo_matrix(Zi),
            "groups": factor.cat.categories.tolist(),
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
        expr_names = self.expr.var_names.copy()
        factor_names = self.factor.var_names.copy()
        return expr_names.union(factor_names)

    def to_string(self, level=None):
        string = ""
        if isinstance(self.expr, Intercept):
            string += "1|"
        elif isinstance(self.expr, Term):
            if level is not None:
                string += f"{self.expr.name}[{level}]|"
            else:
                string += f"{self.expr.name}|"
        else:
            raise ValueError("Invalid LHS expression for group specific term")

        if isinstance(self.factor, Term):
            string += self.factor.name
        else:
            raise ValueError("Invalid RHS expression for group specific term")
        return string


class Response:
    """Representation of a response term"""

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
        # ~ is interpreted as __add__
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
        self.term.set_data(encoding)


ACCEPTED_TERMS = (Term, GroupSpecificTerm, Intercept, NegatedIntercept)


class Model:
    """Representation of the terms in a model"""

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
        """Set union.
        (1 + x + y) + 0 -> (x + y)
        (x + y) + z -> x + y + z
        (x + y) + (u + v) -> x + y + u + v
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
        """Set difference.

        (x + y) - (x + u) -> y + u
        (x + y) - x -> y
        (x + y + (1 | g)) - (1 | g) -> x + y
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

    def __mul__(self, other):
        """Full interaction.

        (x + y) * (u + v) -> x + y + u + v + x:u + x:v + y:u + y:v
        (x + y) * u -> x + y + u + x:u + y:u
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

    def __matmul__(self, other):
        """Simple interaction.

        (x + y) : (u + v) -> x:u + x:v + y:u + y:v
        (x + y) : u -> x:u + y:u
        (x + y) : f(u) -> x:f(u) + y:f(u)
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

    def __pow__(self, other):
        """Power of a set of Terms

        (x + y + z) ** 2 -> x + y + z + x:y + x:z + y:z
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
        """Division interaction.

        See https://patsy.readthedocs.io/en/latest/formulas.html

        (x + y) / z -> x + y + x:y:z
        (x + y) / (u + v) -> x + y + x:y:u + x:y:v
        """
        if isinstance(other, Term):
            return self.add_term(Term(*self.common_components + other.components))
        elif isinstance(other, Model):
            iterms = [Term(*self.common_components, comp) for comp in other.common_components]
            return self + Model(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        """
        Only terms like (0 + x | g) arrive here
        (0 + x | g) -> (x|g)
        (0 + x | g + y) -> (x|g) + (x|y)
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
        """Add response term to model description."""
        if isinstance(term, Response):
            self.response = term
            return self
        else:
            raise ValueError("not Response")

    def add_term(self, term):
        """Add term to model description.

        The term added can be of class ``Intercept``, ``Term``, or ``GroupSpecificTerm``. It appends
        the new term object to the list of common terms or group specific terms as appropriate.
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

        Returns a list of both common and group specific terms in the model description.
        """
        return self.common_terms + self.group_terms

    @property
    def common_components(self):
        """Components in common terms in the model.

        Returns a list with all components, ``Variable`` and ``Call`` instances, within common
        terms in the model.
        """
        # Note: Check whether this method is really necessary.
        return [
            comp for term in self.common_terms if isinstance(term, Term) for comp in term.components
        ]

    @property
    def var_names(self):
        """Returns the name of the variables in the model as a set."""
        var_names = set()
        for term in self.terms:
            var_names.update(term.var_names)
        if self.response is not None:
            var_names.update(self.response.var_names)
        return var_names

    def set_types(self, data, eval_env):
        """Set the type of the common terms in the model."""
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

        This method returns dictionaries with True/False values.
        True means the categorical variable uses 'levels' dummies.
        False means the categorial variable uses 'levels - 1' dummies.
        """
        groups = self._encoding_groups()
        l = [pick_contrasts(group) for group in groups]
        result = dict()
        for d in l:
            result.update(d)
        return result

    def eval(self, data, eval_env):
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


# IDEA: What if Variable, Call, Terms, etc... get frozen once set_type or similar is called?
#       Then, all properties and alike are ensured to remain constant and not change...
#       idk, may need to think about it more.
