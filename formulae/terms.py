import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd

from functools import reduce
from itertools import combinations, product
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype
from scipy import linalg, sparse

from .call_utils import CallEvalPrinter, CallNamePrinter, CallVarsExtractor
from .eval import eval_in_data_mask
from .utils import flatten_list, get_interaction_matrix


class BaseTerm:
    """Base class for terms created to share some common methods"""

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
    """Representation of a (common) term in the model.

    Parameters
    ----------
    name : str
        Name of the term. It can be the name of a variable, or a function of the name(s) of the
        variable(s) involved.
    variables : str or list
        The name of the variable(s) involved in the term.
    kind : str
        An optional type for the Term. Possible values are 'numeric' and 'categoric'.
    """

    def __init__(self, name, variable, level=None, kind=None):
        self.name = name
        self.variable = variable
        if level is not None:
            self.level = level.name.lexeme
        else:
            self.level = level
        self.kind = kind

    def __hash__(self):
        return hash((self.name, self.variable, self.kind))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.name == other.name and self.variable == other.variable and self.kind == other.kind
        )

    def __mul__(self, other):
        if isinstance(other, (Term, InteractionTerm, CallTerm)):
            if self == other:
                # x * x -> x + x + x:x -> x
                return self
            # x * y -> x + y + x:y
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, ModelTerms):
            # x * (y + z) -> x + y + z + x:y + x:z
            products = product([self], other.common_terms)
            terms = [self] + other.common_terms
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Term, CallTerm)):
            if self == other:
                # x:x -> x
                return self
            # x:y -> x:y
            return InteractionTerm(self, other)
        elif isinstance(other, InteractionTerm):
            # x : (y:z) -> x:y:z
            return other.add_term(self)
        elif isinstance(other, ModelTerms):
            products = product([self], other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __pow__(self, other):
        # x ** n -> x
        if isinstance(other, LiteralTerm) and isinstance(other.value, int) and other.value >= 1:
            return self
        else:
            raise ValueError("Power must be a positive integer")

    def __truediv__(self, other):
        # x / y -> x + x:y
        if isinstance(other, (Term, CallTerm)):
            if self == other:
                return self
            return ModelTerms(self, InteractionTerm(self, other))
        elif isinstance(other, InteractionTerm):
            # x / z:y -> x + x:z:y
            return ModelTerms(self, other.add_term(self))
        elif isinstance(other, ModelTerms):
            # x / (z + y) -> x + x:z + x:y
            products = product([self], other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(self) + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            # (x | g) -> (1|g) + (x|g)
            # implicit intercept
            terms = [GroupSpecTerm(InterceptTerm(), other), GroupSpecTerm(self, other)]
            return ModelTerms(*terms)
        elif isinstance(other, ModelTerms):
            # (x | g + h) -> (x|g) + (x|h) -> (1|g) + (1|h) + (x|g) + (x|h)
            iterms = [
                GroupSpecTerm(InterceptTerm(), p[1]) for p in product([self], other.common_terms)
            ]
            terms = [GroupSpecTerm(p[0], p[1]) for p in product([self], other.common_terms)]
            return ModelTerms(*iterms) + ModelTerms(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "name= " + self.name,
            "variable= " + self.variable,
            "kind= " + str(self.kind),
        ]
        if self.level is not None:
            string_list.append("level= " + self.level)
        return "Term(\n  " + "\n  ".join(string_list) + "\n)"

    @property
    def vars(self):
        return self.variable

    def eval(self, data, eval_env, is_response=False):
        # We don't support multiple level categoric responses yet.
        # `is_response` flags whether the term evaluated is response
        # and returns a 1d array of 0-1 encoding instead of a matrix in case there are
        # multiple levels.
        # In the future, we can support multiple levels.
        x = data[self.variable]

        if is_numeric_dtype(x):
            return self.eval_numeric(x)
        elif is_string_dtype(x) or is_categorical_dtype(x):
            return self.eval_categoric(x, is_response)
        else:
            raise NotImplementedError

    def eval_numeric(self, x):
        if self.level is not None:
            raise ValueError("Subset notation can't be used with a numeric variable.")
        out = {"value": np.atleast_2d(x.to_numpy()).T, "type": "numeric"}
        return out

    def eval_categoric(self, x, is_response):
        # If not ordered, we make it ordered
        # x.unique() preservese order of appearence
        if not hasattr(x, "ordered") or not x.ordered:
            cat_type = pd.api.types.CategoricalDtype(categories=x.unique().tolist(), ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if is_response:
            if self.level is not None:
                reference = self.level
            value = np.atleast_2d(np.where(x == reference, 1, 0)).T
        else:
            # .to_numpy() returns 2d array
            value = pd.get_dummies(x, drop_first=True).to_numpy()

        return {"value": value, "type": "categoric", "levels": levels, "reference": reference}


class InteractionTerm(BaseTerm):
    """Representation of an interaction term

    Parameters
    ----------
    terms: list
        list of Terms taking place in the interaction
    """

    def __init__(self, *terms):
        # Maybe we don't need both variables and terms.
        self.variables = set()
        self.terms = []
        for term in terms:
            self.add_term(term)

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.name == other.name
            and self.terms == other.terms
            and self.variables == other.variables
        )

    def __mul__(self, other):
        if isinstance(other, (Term, CallTerm)):
            # x:y * u -> x:y + u + x:y:u
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, InteractionTerm):
            # x:y * u:v -> x:y + u:v + x:y:u:v
            return ModelTerms(self, other, other.add_term(self))
        elif isinstance(other, ModelTerms):
            # x:y * (u + v) -> x:y + u + v + x:y:u + x:y:v
            products = product([self], other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(self) + other + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            # (x:y) : u -> x:y:u
            return self.add_term(other)
        elif isinstance(other, ModelTerms):
            # (x:y) : (u + v) -> x:y:u + x:y:v
            products = product([self], other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            # (x:y | u) -> (1 | u) + (x:y | u)
            terms = [GroupSpecTerm(InterceptTerm(), other), GroupSpecTerm(self, other)]
            return ModelTerms(*terms)
        elif isinstance(other, ModelTerms):
            # (x:y | u + v) -> (1 | u) + (1 | v) + (x:y|u) + (x:y|v)
            iterms = [
                GroupSpecTerm(InterceptTerm(), p[1]) for p in product([self], other.common_terms)
            ]
            terms = [GroupSpecTerm(p[0], p[1]) for p in product([self], other.common_terms)]
            return ModelTerms(*iterms) + ModelTerms(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = ["name= " + self.name, "variables= " + str(self.variables)]
        return "InteractionTerm(\n  " + "\n  ".join(string_list) + "\n)"

    @property
    def name(self):
        return ":".join([term.name for term in self.terms])

    @property
    def vars(self):
        return [term.vars for term in self.terms]

    def add_term(self, term):
        if isinstance(term, Term):
            if term.variable not in self.variables:
                self.terms.append(term)
                self.variables.add(term.variable)
            return self
        elif isinstance(term, CallTerm):
            if term not in self.terms:
                self.terms.append(term)
            return self
        elif isinstance(term, InteractionTerm):
            terms = [term for term in term.terms if term not in self.terms]
            self.terms = self.terms + terms
            self.variables.update(term.variables)
            return self
        else:
            return NotImplemented

    def eval(self, data, eval_env):
        # I'm not very happy with this implementation since we call `.eval()`
        # again on terms that are highly likely to be in the model.
        # But it works and it's fine for now.
        evaluated_terms = {term.name: term.eval(data, eval_env) for term in self.terms}

        value = reduce(
            get_interaction_matrix, [evaluated_terms[k]["value"] for k in evaluated_terms.keys()]
        )
        out = {"value": value, "type": "interaction", "terms": evaluated_terms}
        return out


class LiteralTerm(BaseTerm):
    def __init__(self, value):
        self.value = value
        self.name = str(self.value)

    def __hash__(self):
        return hash((self.value, self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.value == other.value and self.name == other.name

    def __or__(self, other):
        if self.value != 1:
            # (n | g) does not make sense for n != 0 or 1. Case n=0 handled in NegatedIntercept
            raise ValueError("Numeric in LHS of group specific term must be 0, -1, or 1.")
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            # (1|g) -> (1|g); (1|g:h) -> (1|g:h)
            return GroupSpecTerm(self, other)
        elif isinstance(other, ModelTerms):
            # (1 | g + h) -> (1|g) + (1|h)
            products = product([self], other.common_terms)
            terms = [GroupSpecTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            raise ValueError("'factor' must be a single term.")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"LiteralTerm(value={self.value})"

    @property
    def vars(self):
        return ""

    def eval(self, data, eval_env):
        return {"value": np.ones((data.shape[0], 1)) * self.value, "type": "Literal"}


class InterceptTerm(BaseTerm):
    def __init__(self):
        self.name = "Intercept"

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            # (1|g) -> (1|g); (1|g:h) -> (1|g:h)
            return GroupSpecTerm(self, other)
        elif isinstance(other, ModelTerms):
            # (1 | g + h) -> (1|g) + (1|h)
            products = product([self], other.common_terms)
            terms = [GroupSpecTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "InterceptTerm()"

    @property
    def vars(self):
        return ""

    def eval(self, data, eval_env):
        # Only works with DataFrames or Series so far
        return {"value": np.ones((data.shape[0], 1)), "type": "Intercept"}


class NegatedIntercept(BaseTerm):
    def __init__(self):
        self.name = "Intercept"

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    def __or__(self, other):
        raise ValueError("At least include an intercept in '|' operation")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "NegatedIntercept()"

    @property
    def vars(self):
        return ""


class CallTerm(BaseTerm):
    """Representation of a call term"""

    def __init__(self, expr):
        self.callee = expr.callee.name.lexeme
        self.args = expr.args
        self.special = expr.special
        self.name = self.get_name_str()

    def __hash__(self):
        return hash((self.callee, self.name, self.args, self.special))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.callee == other.callee
            and self.name == other.name
            and self.args == other.args
            and self.special == other.special
        )

    def __mul__(self, other):
        if isinstance(other, (Term, InteractionTerm, CallTerm)):
            # f(x)*y -> f(x) + y + f(x):y
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, ModelTerms):
            # f(x)*(y + z) -> f(x) + y + z + f(x):y + f(x):z
            products = product([self], other.common_terms)
            terms = [self] + list(other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Term, CallTerm)):
            # f(x):y -> f(x):y
            return InteractionTerm(self, other)
        elif isinstance(other, InteractionTerm):
            # f(x):y:z -> f(x):y:z
            return other.add_term(self)
        elif isinstance(other, ModelTerms):
            # f(x):(y:z) -> f(x):y:z
            products = product([self], other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, (Term, CallTerm, InteractionTerm)):
            # (f(x) | u) -> (1 | u) + (f(x) | u)
            terms = [GroupSpecTerm(InterceptTerm(), other), GroupSpecTerm(self, other)]
            return ModelTerms(*terms)
        elif isinstance(other, ModelTerms):
            # (f(x) | u + v) -> (1 | u) + (1 | v) + (f(x) | u) + (f(x) | v)
            iterms = [
                GroupSpecTerm(InterceptTerm(), p[1]) for p in product([self], other.common_terms)
            ]
            terms = [GroupSpecTerm(p[0], p[1]) for p in product([self], other.common_terms)]
            return ModelTerms(*iterms) + ModelTerms(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strlist = [
            "call=" + self.name,
            "args=" + "  ".join(str(self.args).splitlines(True)),
            "special=" + str(self.special),
        ]
        return "CallTerm(\n  " + ",\n  ".join(strlist) + "\n)"

    def accept(self, visitor):
        return visitor.visitCallTerm(self)

    def get_eval_str(self, data_cols):
        return CallEvalPrinter(self, data_cols).print()

    def get_name_str(self):
        return CallNamePrinter(self).print()

    @property
    def vars(self):
        return CallVarsExtractor(self).get()

    def eval(self, data, eval_env, is_response=False):
        # Workaround: var names present in 'data' are taken from '__DATA__['col']
        # the rest are left as they are and looked up in the upper namespace
        data_cols = data.columns.tolist()

        x = eval_in_data_mask(self.get_eval_str(data_cols), data, eval_env)
        if isinstance(x, dict):
            return {"value": x["value"], "type": "categoric", "reference": x["reference"]}
        elif is_categorical_dtype(x) or is_string_dtype(x):
            if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
                cat_type = pd.api.types.CategoricalDtype(
                    categories=x.unique().tolist(), ordered=True
                )
                x = x.astype(cat_type)
            levels = x.cat.categories.tolist()
            value = pd.get_dummies(x, drop_first=True).to_numpy()
            return {"value": value, "type": "categoric", "levels": levels}
        else:
            return {"value": np.atleast_2d(x.to_numpy()).T, "type": "call"}


class GroupSpecTerm(BaseTerm):
    """Representation of group specific effects term"""

    def __init__(self, expr, factor):
        """
        if isinstance(expr, ModelTerms):
            # This happens when we use `0 + x | g1`
            self.expr = expr.terms[0]
        else:
            self.expr = expr
        """
        self.expr = expr
        self.factor = factor

    def __hash__(self):
        return hash((self.expr, self.factor))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.expr == other.expr and self.factor == other.factor

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strlist = [
            "expr= " + "  ".join(str(self.expr).splitlines(True)),
            "factor= " + "  ".join(str(self.factor).splitlines(True)),
        ]
        return "GroupSpecTerm(\n  " + ",\n  ".join(strlist) + "\n)"

    def to_string(self):
        string = ""
        if isinstance(self.expr, InterceptTerm):
            string += "1|"
        elif isinstance(self.expr, (Term, CallTerm)):
            string += self.expr.name + "|"
        else:
            raise ValueError("Invalid LHS expression for group specific term")

        if isinstance(self.factor, Term):
            string += self.factor.name
        else:
            raise ValueError("Invalid RHS expression for group specific term")

        return string

    @property
    def vars(self):
        return [self.expr.vars] + [self.factor.vars]

    def eval(self, data, eval_env):
        if isinstance(self.factor, Term):
            factor = data[self.factor.variable]
            if is_categorical_dtype(factor) or is_string_dtype(factor):
                if not hasattr(factor.dtype, "ordered") or not factor.dtype.ordered:
                    cat_type = pd.api.types.CategoricalDtype(
                        categories=factor.unique().tolist(), ordered=True
                    )
                    factor = factor.astype(cat_type)
        else:
            raise ValueError(
                "Factor on right hand side of group specific term must be a single term."
            )

        # Notation as in lme4 paper
        Ji = pd.get_dummies(factor).to_numpy()  # note we don't use `drop_first=True`.
        Xi = self.expr.eval(data, eval_env)
        Zi = linalg.khatri_rao(Ji.T, Xi["value"].T).T
        out = {
            "type": Xi["type"],
            "Zi": sparse.coo_matrix(Zi),
            "groups": factor.cat.categories.tolist(),
        }
        if Xi["type"] == "categoric":
            out["levels"] = Xi["levels"]
            out["reference"] = Xi["reference"]
        return out


class ResponseTerm:
    """Representation of a response term"""

    def __init__(self, term):
        if isinstance(term, (Term, CallTerm)):
            self.term = term
        else:
            raise ValueError("Response Term must be univariate")

    def __hash__(self):
        return hash((self.term))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.term == other.term

    def __add__(self, other):
        # ~ is interpreted as __add__
        if isinstance(other, ATOMIC_TERMS):
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

    def eval(self, data, eval_env):
        return self.term.eval(data, eval_env, is_response=True)


class ModelTerms:
    """Representation of the terms in a model"""

    def __init__(self, *terms, response=None):
        if isinstance(response, ResponseTerm) or response is None:
            self.response = response
        else:
            raise ValueError("Response must be of class ResponseTerm.")

        if all([isinstance(term, ATOMIC_TERMS) for term in terms]):
            self.common_terms = [term for term in terms if not isinstance(term, GroupSpecTerm)]
            self.group_terms = [term for term in terms if isinstance(term, GroupSpecTerm)]
        else:
            raise ValueError("There is a least one term of an unexpected class.")

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        equal_terms = all([t1 == t2 for t1, t2 in zip(self.terms, other.terms)])
        equal_responses = self.response == other.response
        return equal_terms and equal_responses

    def __add__(self, other):
        # (1 + x + y) + 0 -> (x + y)
        if isinstance(other, NegatedIntercept):
            return self - InterceptTerm()
        elif isinstance(other, ATOMIC_TERMS):
            # (x + y) + z -> x + y + z
            return self.add_term(other)
        elif isinstance(other, type(self)):
            # (x + y) + (u + v) -> x + y + u + v
            for term in other.terms:
                self.add_term(term)
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            # (x + y) - (x + u) -> y + u
            for term in other.terms:
                if term in self.common_terms:
                    self.common_terms.remove(term)
                if term in self.group_terms:
                    self.group_terms.remove(term)
            return self
        elif isinstance(other, (Term, CallTerm, InteractionTerm, InterceptTerm, LiteralTerm)):
            # (x + y) - x -> y
            if other in self.common_terms:
                self.common_terms.remove(other)
            return self
        elif isinstance(other, GroupSpecTerm):
            # (x + y + (1 | g)) - (1 | g) -> x + y
            if other in self.group_terms:
                self.group_terms.remove(other)
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, type(self)):
            # (x + y) * (u + v) -> x + y + u + v + x:u + x:v + y:u + y:v
            products = product(self.common_terms, other.common_terms)
            terms = list(self.common_terms) + list(other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        elif isinstance(other, (Term, CallTerm)):
            # (x + y) * u -> x + y + u + x:u + y:u
            products = product(self.common_terms, [other])
            terms = [term for term in self.common_terms] + [other]
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            # (x + y) : (u + v) -> x:u + x:v + y:u + y:v
            products = product(self.common_terms, other.common_terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        elif isinstance(other, (Term, CallTerm)):
            # (x + y) : u -> x:u + y:u
            products = product(self.common_terms, [other])
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*iterms)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, LiteralTerm) and isinstance(other.value, int) and other.value >= 1:
            # (x + y + z) ** 2 -> x + y + z + x:y + x:z + y:z
            comb = [
                list(p)
                for i in range(2, other.value + 1)
                for p in combinations(self.common_terms, i)
            ]
            iterms = [InteractionTerm(*terms) for terms in comb]
            return self + ModelTerms(*iterms)
        else:
            raise ValueError("Power must be a positive integer")

    def __truediv__(self, other):
        # See https://patsy.readthedocs.io/en/latest/formulas.html
        if isinstance(other, (Term, CallTerm)):
            # (x + y) / z -> x + y + x:y:z
            return self.add_term(InteractionTerm(*self.common_terms + [other]))
        elif isinstance(other, ModelTerms):
            # (x + y) / (u + v) -> x + y + x:y:u + x:y:v
            iterms = [InteractionTerm(*self.common_terms + [term]) for term in other.common_terms]
            return self + ModelTerms(*iterms)
        else:
            return NotImplemented

    def __or__(self, other):
        # Only negated intercept + one term
        if NegatedIntercept() in self.common_terms and len(self.common_terms) == 2:
            self.common_terms.remove(NegatedIntercept())
            if isinstance(other, (Term, CallTerm, InteractionTerm)):
                # (0 + x | g) -> (x|g)
                return GroupSpecTerm(self.common_terms[0], other)
            elif isinstance(other, ModelTerms):
                # (0 + x | g + y) -> (x|g) + (x|y)
                terms = [GroupSpecTerm(p[0], p[1]) for p in product([self], other.common_terms)]
                return ModelTerms(*terms)
            else:
                return NotImplemented
        else:
            raise ValueError("LHS of group specific term cannot have more than one term.")

    def __repr__(self):
        terms = ",\n".join([repr(term) for term in self.common_terms])
        if self.response is None:
            string = "  ".join(terms.splitlines(True))
        else:
            string = "  ".join(str(self.response).splitlines(True))
            string += ",\n  " + "  ".join(terms.splitlines(True))

        if self.group_terms:
            group_terms = ",\n".join([repr(term) for term in self.group_terms])
            string += ",\n  " + "  ".join(group_terms.splitlines(True))

        return "ModelTerms(\n  " + string + "\n)"

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
        elif isinstance(term, ATOMIC_TERMS):
            if term not in self.common_terms:
                self.common_terms.append(term)
            return self
        else:
            raise ValueError("not accepted term")

    @property
    def terms(self):
        return self.common_terms + self.group_terms

    @property
    def vars(self):
        vars = set()
        for term in self.terms:
            if isinstance(term, (CallTerm, InteractionTerm, GroupSpecTerm)):
                v = term.vars
                if isinstance(v, list):
                    # make sure it is flattened
                    v = flatten_list(v)
                vars = vars.union(set(v))
            else:
                vars = vars.union({term.vars})
        if self.response is not None:
            vars = vars.union({self.response.vars})

        # Some terms return '' for vars
        vars = vars - {""}
        return vars

    def eval(self, data, eval_env):
        return {term.name: term.eval(data, eval_env) for term in self.terms}


ATOMIC_TERMS = (
    Term,
    InteractionTerm,
    CallTerm,
    InterceptTerm,
    NegatedIntercept,
    LiteralTerm,
    GroupSpecTerm,
)
