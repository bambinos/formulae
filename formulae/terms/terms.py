import numpy as np

from functools import reduce
from itertools import combinations, product

from formulae.utils import get_interaction_matrix


class Intercept:
    def __init__(self):
        self.name = "Intercept"
        self._type = "Intercept"

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __add__(self, other):
        if isinstance(other, NegatedIntercept):
            return ModelTerms()
        elif isinstance(other, type(self)):
            return self
        elif isinstance(other, (Term, GroupSpecTerm)):
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return ModelTerms(self) + other
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            if self.components == other.components:
                return ModelTerms()
            else:
                return self
        elif isinstance(other, ModelTerms):
            if self in other.common_terms:
                return ModelTerms()
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
            return GroupSpecTerm(self, other)
        elif isinstance(other, ModelTerms):
            products = product([self], other.common_terms)
            terms = [GroupSpecTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}()"

    @property
    def vars(self):
        return ""

    def eval(self, data, eval_env, encoding):
        # Only works with DataFrames or Series so far
        return {"value": np.ones((data.shape[0], 1)), "type": "Intercept"}

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
            return ModelTerms()
        elif isinstance(other, (Term, GroupSpecTerm)):
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return ModelTerms(self) + other
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
    def vars(self):
        return ""

class Term:
    """Representation of a single term.

    A model term can be an intercept, a term made of a single component, a function call,
    an interaction involving components and/or function calls.
    """

    def __init__(self, *components):
        self.components = []
        for component in components:
            if component not in self.components:
                self.components.append(component)

    def __hash__(self):
        return hash(*self.components)

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
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return ModelTerms(self) + other
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
                return ModelTerms()
            else:
                return self
        elif isinstance(other, ModelTerms):
            if self in other.terms:
                return ModelTerms()
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
            return ModelTerms(self, other, Term(*self.components, *other.components))
        elif isinstance(other, ModelTerms):
            products = product([self], other.common_terms)
            terms = [self] + other.common_terms
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
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
            return Term(*self.components, *other.components)
        elif isinstance(other, ModelTerms):
            products = product([self], other.common_terms)
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __pow__(self, other):
        """Power of a ``Term``.

        It leaves the term as it is. For a power in the math sense do ``I(x ** n)`` or ``{x ** n}``.
        """
        if len(other.components) == 1:
            expr = other.components[0].expr
            if isinstance(expr, int) and expr >= 1:
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
            return ModelTerms(self, Term(*self.components, *other.components))
        elif isinstance(other, ModelTerms):
            products = product([self], other.common_terms)
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return self + ModelTerms(*iterms)
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
            terms = [GroupSpecTerm(Intercept(), other), GroupSpecTerm(self, other)]
            return ModelTerms(*terms)
        elif isinstance(other, ModelTerms):
            intercepts = [
                GroupSpecTerm(Intercept(), p[1])
                 for p in product([self], other.common_terms)
            ]
            slopes = [GroupSpecTerm(p[0], p[1]) for p in product([self], other.common_terms)]
            return ModelTerms(*intercepts, *slopes)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "[" + ", ".join([repr(component) for component in self.components]) + "]"
        return f"{self.__class__.__name__}({string})"

    def eval(self, data, eval_env, encoding):
        # TODO: Clean this implementation

        if isinstance(encoding, list) and len(encoding) == 1:
            encoding = encoding[0]
        else:
            ValueError("encoding is a list of len > 1")

        evaluated_terms = dict()
        for component in self.components:
            encoding_ = []
            # encoding is emtpy list when all numerics
            if isinstance(encoding, dict):
                if component.name in encoding.keys():
                    encoding_ = encoding[component.name]
            # Set type.
            # Set data.
            # And then extract that data.
            evaluated_terms[component.name] = component.eval(data, eval_env, encoding_)

        value = reduce(
            get_interaction_matrix, [evaluated_terms[k]["value"] for k in evaluated_terms.keys()]
        )
        return {"value": value, "type": "interaction", "terms": evaluated_terms}


class GroupSpecTerm:
    def __init__(self, expr, factor):
        self.expr = expr
        self.factor = factor

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.expr == other.expr and self.factor == other.factor

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strlist = [
            f"expr= {'  '.join(str(self.expr).splitlines(True))}",
            f"factor= {'  '.join(str(self.factor).splitlines(True))}",
        ]
        return self.__class__.__name__ + "(\n  " + ',\n  '.join(strlist) + "\n)"

class ResponseTerm:
    """Representation of a response term"""

    def __init__(self, term):
        if isinstance(term, Term):
            # Check term is a unique term or a call, and not interaction or something different
            self.term = term
        else:
            raise ValueError("Response Term must be univariate")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.term == other.term

    def __add__(self, other):
        # ~ is interpreted as __add__
        if isinstance(other, (Term, Intercept)):
            return ModelTerms(other, response=self)
        elif isinstance(other, ModelTerms):
            return other.add_response(self)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "ResponseTerm(\n  " + "  ".join(str(self.term).splitlines(True)) + "\n)"

    @property
    def vars(self):
        return self.term.vars

    def eval(self, data, eval_env, encoding=None):
        return self.term.eval(data, eval_env, encoding, is_response=True)

ACCEPTED_TERMS = (Term, GroupSpecTerm, Intercept, NegatedIntercept)

class ModelTerms:
    """Representation of the terms in a model"""

    def __init__(self, *terms, response=None):
        if isinstance(response, ResponseTerm) or response is None:
            self.response = response
        else:
            raise ValueError("Response must be of class ResponseTerm.")
        if all(isinstance(term, ACCEPTED_TERMS) for term in terms):
            self.common_terms = [term for term in terms if not isinstance(term, GroupSpecTerm)]
            self.group_terms = [term for term in terms if isinstance(term, GroupSpecTerm)]
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
        elif isinstance(other, (Term, GroupSpecTerm, Intercept)):
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
        elif isinstance(other, GroupSpecTerm):
            if other in self.group_terms:
                self.group_terms.remove(other)
            return self
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
            return ModelTerms(*iterms)
        elif isinstance(other, Term):
            products = product(self.common_terms, [other])
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __pow__(self, other):
        """Power of a set of Terms

        (x + y + z) ** 2 -> x + y + z + x:y + x:z + y:z
        """

        if isinstance(other, Term) and len(other.components) == 1:
            expr = other.components[0].expr
            if isinstance(expr, int) and expr >= 1:
                comb = [
                    list(p)
                    for i in range(2, expr + 1)
                    for p in combinations(self.common_terms, i)
                ]
            iterms = [Term(*[comp for term in terms for comp in term.components]) for terms in comb]
            return self + ModelTerms(*iterms)
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
        elif isinstance(other, ModelTerms):
            iterms = [Term(*self.common_components, comp) for comp in other.common_components]
            return self + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        """
        Only terms like (0 + x | g) arrive here
        (0 + x | g) -> (x|g)
        (0 + x | g + y) -> (x|g) + (x|y)
        """
        # Only negated intercept + one term
        if len(self.common_terms) <= 2:
            if len(self.common_terms) == 1:
                return self.common_terms[0] | other
            if NegatedIntercept() in self.common_terms:
                self.common_terms.remove(NegatedIntercept())

            if isinstance(other, Term):
                products = product(self.common_terms, [other])
                terms = [GroupSpecTerm(p[0], p[1]) for p in products]
                return ModelTerms(*terms)
            elif isinstance(other, type(self)):
                products = product(self.common_terms, other.common_terms)
                terms = [GroupSpecTerm(p[0], p[1]) for p in products]
                return ModelTerms(*terms)
            else:
                return NotImplemented
        else:
            raise ValueError("LHS of group specific term cannot have more than one term.")

    def add_response(self, term):
        if isinstance(term, ResponseTerm):
            self.response = term
            return self
        else:
            raise ValueError("not ResponseTerm")

    def add_term(self, term):
        if isinstance(term, GroupSpecTerm):
            if term not in self.group_terms:
                self.group_terms.append(term)
            return self
        elif isinstance(term, (Term, Intercept)):
            if term not in self.common_terms:
                self.common_terms.append(term)
            return self
        else:
            raise ValueError("Not accepted term.")

    @property
    def terms(self):
        return self.common_terms + self.group_terms

    @property
    def common_components(self):
        return [comp for term in self.common_terms for comp in term.components]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        terms = ",\n".join([repr(term) for term in self.common_terms])
        string = "  ".join(terms.splitlines(True))

        if self.group_terms:
            group_terms = ",\n".join([repr(term) for term in self.group_terms])
            if len(string) > 0:
                string += ",\n  "
            string += "  ".join(group_terms.splitlines(True))

        return f"{self.__class__.__name__}(\n  {string}\n)"