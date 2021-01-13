import numpy as np
import pandas as pd

from itertools import product, combinations
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype

from .eval_in_data_mask import eval_in_data_mask

class BaseTerm:
    """Base Class created to share some common methods
    """

    def __init__(self):
        pass

    def __add__(self, other):
        if isinstance(other, NegatedIntercept):
            return ModelTerms(self) - InterceptTerm()
        if isinstance(other, ATOMIC_TERMS):
            if self == other:
                return self
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return ModelTerms(self) + other
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (ATOMIC_TERMS, ModelTerms)):
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        return NotImplemented

    def __matmul__(self, other):
        return NotImplemented

    def __pow__(self, other):
        return NotImplemented

    def __truediv__(self, other):
        return NotImplemented

    def __or__(self, other):
        return NotImplemented

class Term(BaseTerm):
    """Representation of a model term.

    Parameters
    ----------
    name : str
        Name of the term. It can be the name of a variable, or a function of the name(s) of the
        variable(s) involved.
    variables : str or list
        The name of the variable(s) involved in the term.
    kind : str
        An optional type for the Term. Possible values are 'numeric' and 'categoric'.
        TODO: Add kind 'ordinal'.
    """

    def __init__(self, name, variable, kind=None):
        self.name = name
        self.variable = variable
        self.kind = kind

    def __hash__(self):
        return hash((self.name, self.variable, self.kind))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.name == other.name and self.variable == other.variable and self.kind == other.kind

    def __mul__(self, other):
        if isinstance(other, (Term, CallTerm)):
            if self == other:
                return self
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, InteractionTerm):
            return ModelTerms(self, other, other.add_term(self))
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Term, CallTerm)):
            if self == other:
                return self
            return InteractionTerm(self, other)
        if isinstance(other, InteractionTerm):
            return other.add_term(self)

    def __pow__(self, other):
        if isinstance(other, LiteralTerm) and isinstance(other.value, int) and other.value >= 1:
            return self
        else:
            raise ValueError("Power must be a positive integer")

    def __truediv__(self, other):
        if isinstance(other, (Term, CallTerm)):
            if self == other:
                return self
            return ModelTerms(self, InteractionTerm(self, other))
        elif isinstance(other, InteractionTerm):
            return ModelTerms(self, other.add_term(self))
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(self) + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            # implicit intercept
            terms = [RandomTerm(InterceptTerm(), other), RandomTerm(self, other)]
            return ModelTerms(*terms)
        elif isinstance(other, ModelTerms):
            iterms = [RandomTerm(InterceptTerm(), p[1]) for p in product([self], other.fixed_terms)]
            terms = [RandomTerm(p[0], p[1]) for p in product([self], other.fixed_terms)]
            return ModelTerms(*iterms) + ModelTerms(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "name= " + self.name,
            "variable= " + self.variable,
            "kind= " + str(self.kind)
        ]
        return 'Term(\n  ' + '\n  '.join(string_list) + '\n)'

    def eval(self, data):
        x = data[self.variable]

        if is_numeric_dtype(x):
            return self.eval_numeric(x)
        elif is_string_dtype(x) or is_categorical_dtype(x):
            return self.eval_categoric(x)
        else:
            return NotImplemented

    def eval_numeric(self, x):
        # TODO: Return column numbers
        out = {
            'value': np.atleast_2d(x.to_numpy()).T,
            'type': 'numeric'
        }
        return out

    def eval_categoric(self, x, full_rank=True):
        if isinstance(x, pd.Categorical) and x.dtype.ordered:
            reference = x.min()
        else:
            reference = x[0]

        if full_rank:
            # .to_numpy() returns 2d array
            value = pd.get_dummies(x, drop_first=True).to_numpy()
        else:
            raise NotImplementedError

        out = {
            'value': value,
            'type': 'categoric',
            'reference': reference
        }
        return out

class InteractionTerm(BaseTerm):
    """Representation of an interaction term

    Parameters
    ----------
    name : str
        Name of the term.
    terms: list
        list of Terms taking place in the interaction
    """

    def __init__(self, *terms):
        # self.terms is a list because I have to admit repeated terms -> why?
        # self.variables is a set because i want to store each variable once
        # but there must be a better way to do this
        self.variables = set()
        self.terms = []
        for term in terms:
            self.add_term(term)

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.name == other.name and self.terms == other.terms and self.variables == other.variables

    def __mul__(self, other):
        if isinstance(other, (Term, CallTerm)):
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, InteractionTerm):
            return ModelTerms(self, other, other.add_term(self))
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            return self.add_term(other)
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            return RandomTerm(self, other)
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            terms = [RandomTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "name= " + self.name,
            "variables= " + str(self.variables)
        ]
        return 'InteractionTerm(\n  ' + '\n  '.join(string_list) + '\n)'

    @property
    def name(self):
        return ":".join([term.name for term in self.terms])

    def add_term(self, term):
        if isinstance(term, Term):
            if term.variable not in self.variables:
                self.terms.append(term)
                self.variables.add(term.variable)
            return self
        elif isinstance(term, CallTerm):
            if term.call not in self.variables:
                self.terms.append(term)
                self.variables.add(term.call)
            return self
        elif isinstance(term, InteractionTerm):
            terms = [term for term in term.terms if term not in self.terms]
            self.terms = self.terms + terms
            self.variables.update(term.variables)
            return self
        else:
            return NotImplemented

class LiteralTerm(BaseTerm):
    def __init__(self, value):
        self.value = value
        self.name = str(self.value)

    def __hash__(self):
        return hash((self.value, self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.value == other.value and self.name == other.name

    def __or__(self, other):
        if self.value != 1:
            raise ValueError("Numeric expression is not equal to 1 in random term.")
        if isinstance(other, (Term, CallTerm)):
            return RandomTerm(self, other)
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            terms = [RandomTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            raise ValueError("'factor' must be a single term.")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"LiteralTerm(value={self.value})"

class InterceptTerm(BaseTerm):
    def __init__(self):
        self.name = "Intercept"

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.name == other.name

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            return RandomTerm(self, other)
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            terms = [RandomTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "InterceptTerm()"

    def eval(self, data):
        # Only works with DataFrames or Series so far
        out = {
            'value': np.ones((data.shape[0], 1)),
            'type': 'Intercept'
        }
        return out


class NegatedIntercept(BaseTerm):
    def __init__(self):
        self.name = "Intercept"

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.name == other.name

    def __or__(self, other):
        raise ValueError("At least include an intercept in '|' operation")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "NegatedIntercept()"

class CallTerm(BaseTerm):
    """Representation of a call term
    """

    def __init__(self, expr):
        # self.call = expr.callee.name.lexeme + expr.args.lexeme
        self.name = expr.callee.name.lexeme
        self.args = expr.args
        self.special = expr.special

    def __hash__(self):
        return hash((self.name, self.args, self.special))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.name == other.name, self.args == other.args and self.special == other.special

    def __mul__(self, other):
        if isinstance(other, (Term, InteractionTerm, CallTerm, LiteralTerm)):
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            terms = [self] + list(other.fixed_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Term, CallTerm, LiteralTerm)):
            return InteractionTerm(self, other)
        elif isinstance(other, InteractionTerm):
            return other.add_term(self)
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            return RandomTerm(self, other)
        elif isinstance(other, ModelTerms):
            products = product([self], other.fixed_terms)
            terms = [RandomTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strlist = [
            "call=" + self.name,
            "args=" + '  '.join(str(self.args).splitlines(True)),
            "special=" + str(self.special)
        ]
        return 'CallTerm(\n  ' + ',\n  '.join(strlist) + '\n)'

class RandomTerm(BaseTerm):
    """Representation of random effects term
    """

    def __init__(self, expr, factor):
        # 'expr' and 'factor' names are taken from lme4
        self.expr = expr
        self.factor = factor

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strlist = [
            "expr= " + '  '.join(str(self.expr).splitlines(True)),
            "factor= " + '  '.join(str(self.factor).splitlines(True)),
        ]

        return 'RandomTerm(\n  ' + ',\n  '.join(strlist) + '\n)'

class ResponseTerm:
    """Representation of a response term
    """

    # TODO: things like {x - y} must be a Term and not ModelTerms
    def __init__(self, term):
        if isinstance(term, (Term, CallTerm)):
            self.term = term
        else:
            raise ValueError("Response Term must be univariate")

    def __hash__(self):
        return hash((self.term))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return False
        return self.term == other.term

    def __add__(self, other):
        if isinstance(other, ATOMIC_TERMS):
            return ModelTerms(other, response=self)
        elif isinstance(other, ModelTerms):
            return other.add_response(self)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'ResponseTerm(\n  ' + str(self.term) + '\n)'


class ModelTerms:

    def __init__(self, *terms, response=None):
        if isinstance(response, ResponseTerm) or response is None:
            self.response = response
        else:
            raise ValueError("bad ResponseTerm")

        if all([isinstance(term, ATOMIC_TERMS) for term in terms]):
            self.fixed_terms = [term for term in terms if not isinstance(term, RandomTerm)]
            self.random_terms = [term for term in terms if isinstance(term, RandomTerm)]
        else:
            raise ValueError("Can't understand Term")

    def __add__(self, other):
        if isinstance(other, NegatedIntercept):
            return self - InterceptTerm()
        elif isinstance(other, ATOMIC_TERMS):
            return self.add_term(other)
        elif isinstance(other, type(self)):
            for term in other.all_terms:
                self.add_term(term)
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            for term in other.all_terms:
                if term in self.fixed_terms:
                    self.fixed_terms.remove(term)
                if term in self.random_terms:
                    self.random_terms.remove(term)
            return self
        elif isinstance(other, (Term, CallTerm, InteractionTerm, InterceptTerm)):
            if other in self.fixed_terms:
                self.fixed_terms.remove(other)
            return self
        elif isinstance(other, RandomTerm):
            if other in self.random_terms:
                self.random_terms.remove(other)
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, type(self)):
            products = product(self.fixed_terms, other.fixed_terms)
            terms = list(self.fixed_terms) + list(other.fixed_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        elif isinstance(other, (Term, CallTerm)):
            products = product(self.fixed_terms, [other])
            terms = [term for term in self.fixed_terms] + [other]
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            products = product(self.fixed_terms, other.fixed_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        elif isinstance(other, (Term, CallTerm)):
            products = product(self.fixed_terms, [other])
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, LiteralTerm) and isinstance(other.value, int) and other.value >= 1:
            comb = [list(p) for i in range(2, other.value + 1) for p in combinations(self.fixed_terms, i)]
            iterms = [InteractionTerm(*terms) for terms in comb]
            return self + ModelTerms(*iterms)
        else:
            raise ValueError("Power must be a positive integer")

    def __truediv__(self, other):
        # See https://patsy.readthedocs.io/en/latest/formulas.html
        if isinstance(other, (Term, CallTerm)):
            return self.add_term(InteractionTerm(*self.fixed_terms + [other]))
        elif isinstance(other, ModelTerms):
            iterms = [InteractionTerm(*self.fixed_terms + [term]) for term in other.fixed_terms]
            return self + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            if NegatedIntercept() in self.fixed_terms:
                self.fixed_terms.remove(NegatedIntercept())
                return RandomTerm(self, other)
            else:
                terms = InterceptTerm() + self
                return RandomTerm(terms, other)
        else:
            return NotImplemented

    def __repr__(self):
        terms = ',\n'.join([repr(term) for term in self.fixed_terms])
        if self.response is None:
            string = '  '.join(terms.splitlines(True))
        else:
            string = '  '.join(str(self.response).splitlines(True))
            string += ',\n  ' + '  '.join(terms.splitlines(True))

        if self.random_terms:
            random_terms = ',\n'.join([repr(term) for term in self.random_terms])
            string += ',\n  ' + '  '.join(random_terms.splitlines(True))

        return 'ModelTerms(\n  ' + string + '\n)'

    def add_response(self, term):
        if isinstance(term, ResponseTerm):
            self.response = term
            return self
        else:
            raise ValueError("not ResponseTerm")

    def add_term(self, term):
        if isinstance(term, RandomTerm):
            if term not in self.random_terms:
                self.random_terms.append(term)
            return self
        elif isinstance(term, ATOMIC_TERMS):
            if term not in self.fixed_terms:
                self.fixed_terms.append(term)
            return self
        else:
            raise ValueError("not accepted term")

    @property
    def terms(self):
        return self.fixed_terms + self.random_terms

    def eval(self, data):
        return {term.name:term.eval(data) for term in self.terms}
        # return np.vstack([term.eval(data) for term in self.terms]).T

ATOMIC_TERMS = (Term, InteractionTerm, CallTerm, InterceptTerm, NegatedIntercept, LiteralTerm, RandomTerm)