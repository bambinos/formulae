from formulae.expr import Literal
from itertools import combinations, product

class Variable:
    """Atomic component of a Term"""

    def __init__(self, expr):
        self.expr = expr

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.expr == other.expr

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}({self.expr})"

class Term:
    """Representation of a single term in a ModelTerms.

    A model term can be an intercept, a single component, a function call, an interaction
    involving components and/or function calls or a group specific term.
    """

    def __init__(self, *components, is_call=False):
        self.components = []
        self.is_call = is_call
        for component in components:
            if component not in self.components:
                self.components.append(component)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        else:
            return self.components == other.components

    def __add__(self, other):
        if self == other:
            return self
        elif isinstance(other, type(self)):
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return ModelTerms(self) + other
        else:
            raise ValueError("Unsupported RHS in '+' operation.")

    def __sub__(self, other):
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
            raise ValueError("Unsupported RHS in '-' operation.")

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
            raise ValueError("Unsupported RHS in '*' operation.")

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
            raise ValueError("Unsupported RHS in ':' operation.")

    def __pow__(self, other):
        """Power of a Term.

        It leaves the term as it is. For a mathematical power, do `I(x ** n)` or `{x ** n}`.
        """
        if len(other.components) == 1:
            expr = other.components[0].expr
            if isinstance(expr, int) and expr >= 1:
                return self
        else:
            raise ValueError("Power must be a positive integer.")

    def __truediv__(self, other):
        """Division interaction.

        x / x -> x
        x / y -> x + x:y
        x / z:y -> x + x:z:y
        x / (z + y) -> x + x:z + x:y
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
            raise ValueError("Unsupported RHS in '/' operation.")

    def __or__(self, other):
        """Group specific term

        (x|g) -> (1|g) + (x|g)
        (x|g + h) -> (x|g) + (x|h) -> (1|g) + (1|h) + (x|g) + (x|h)
        """
        if isinstance(other, Term):
            # Only accepts terms, call terms and interactions.
            # Adds implicit intercept.
            terms = [GroupSpecTerm(Term(Variable([])), other), GroupSpecTerm(self, other)]
            return ModelTerms(*terms)
        elif isinstance(other, ModelTerms):
            intercepts = [
                GroupSpecTerm(Term(Variable([])), p[1])
                 for p in product([self], other.common_terms)
            ]
            slopes = [GroupSpecTerm(p[0], p[1]) for p in product([self], other.common_terms)]
            return ModelTerms(*intercepts, *slopes)
        else:
            raise ValueError("Unsupported RHS in '|' operation.")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "[" + ", ".join([repr(component) for component in self.components]) + "]"
        return f"{self.__class__.__name__}({string})"


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
        if isinstance(other, Term):
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

class ModelTerms:
    """Representation of the terms in a model"""

    def __init__(self, *terms, response=None):
        if isinstance(response, ResponseTerm) or response is None:
            self.response = response
        else:
            raise ValueError("Response must be of class ResponseTerm.")

        if all(isinstance(term, (Term, GroupSpecTerm)) for term in terms):
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
        """
        if isinstance(other, Term):
            return self.add_term(other)
        elif isinstance(other, type(self)):
            for term in other.terms:
                self.add_term(term)
            return self
        else:
            raise ValueError("Unsupported RHS in '+' operation.")

    def __sub__(self, other):
        """Set difference.
        """
        if isinstance(other, type(self)):
            for term in other.terms:
                if term in self.common_terms:
                    self.common_terms.remove(term)
                if term in self.group_terms:
                    self.group_terms.remove(term)
            return self
        elif isinstance(other, Term):
            if other in self.common_terms:
                self.common_terms.remove(other)
            return self
        elif isinstance(other, GroupSpecTerm):
            if other in self.group_terms:
                self.group_terms.remove(other)
            return self
        else:
            raise ValueError("Unsupported RHS in '-' operation.")

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
            raise ValueError("Unsupported RHS in ':' operation.")

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
            raise ValueError("Unsupported RHS in '/' operation.")

    def __or__(self, other):
        """
        (0 + x | g) -> (x|g)
        (0 + x | g + y) -> (x|g) + (x|y)
        """
        if len(self.common_terms) <= 2:
            if isinstance(other, Term):
                products = product(self.common_terms, [other])
                terms = [GroupSpecTerm(p[0], p[1]) for p in products]
                return ModelTerms(*terms)
            elif isinstance(other, type(self)):
                products = product(self.common_terms, other.common_terms)
                terms = [GroupSpecTerm(p[0], p[1]) for p in products]
                return ModelTerms(*terms)
            else:
                raise ValueError("Unsupported RHS in '|' operation.")
        else:
            # If more than two terms, raise error
            raise ValueError(
                "LHS of group specific term cannot have more than intercept and one term."
            )

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
        elif isinstance(term, Term):
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