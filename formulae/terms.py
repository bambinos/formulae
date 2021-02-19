import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd

from functools import reduce
from itertools import combinations, product
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype
from scipy import linalg, sparse

from .call_utils import CallEvalPrinter, CallNamePrinter, CallVarsExtractor
from .contrasts import pick_contrasts
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

    def components(self, data, eval_env):
        # Returns components and whether they are categoric or numeric
        x = data[self.variable]
        if is_numeric_dtype(x):
            type_ = "numeric"
        elif is_string_dtype(x) or is_categorical_dtype(x):
            type_ = "categoric"
        else:
            raise NotImplementedError
        return {self.name: type_}

    def eval(self, data, eval_env, encoding, is_response=False):
        # We don't support multiple level categoric responses yet.
        # `is_response` flags whether the term evaluated is response
        # and returns a 1d array of 0-1 encoding instead of a matrix in case there are
        # multiple levels.
        # In the future, we can support multiple levels.
        x = data[self.variable]

        if is_numeric_dtype(x):  # If numerical, evaluate as it is, no encoding needed.
            return self.eval_numeric(x)
        elif is_string_dtype(x) or is_categorical_dtype(x):
            return self.eval_categoric(x, encoding, is_response)
        else:
            raise NotImplementedError

    def eval_numeric(self, x):
        if self.level is not None:
            raise ValueError("Subset notation can't be used with a numeric variable.")
        out = {"value": np.atleast_2d(x.to_numpy()).T, "type": "numeric"}
        return out

    def eval_categoric(self, x, encoding, is_response):
        # If not ordered, we make it ordered
        # x.unique() preservese order of appearence

        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            cat_type = pd.api.types.CategoricalDtype(categories=x.unique().tolist(), ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if is_response:
            if self.level is not None:
                reference = self.level
            value = np.atleast_2d(np.where(x == reference, 1, 0)).T
        else:
            if isinstance(encoding, list):
                encoding = encoding[0]
            if isinstance(encoding, dict):
                encoding = encoding[self.name]
            if encoding:
                value = pd.get_dummies(x).to_numpy()
                encoding = "full"
            else:
                value = pd.get_dummies(x, drop_first=True).to_numpy()
                encoding = "reduced"
        return {
            "value": value,
            "type": "categoric",
            "levels": levels,
            "reference": reference,
            "encoding": encoding,
        }


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

    def get_term(self, name):
        idx = [t.name == name for t in self.terms].index(True)
        return self.terms[idx]

    def components(self, data, eval_env):
        return {term.name: term.components(data, eval_env)[term.name] for term in self.terms}

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

    def eval(self, data, eval_env, encoding):
        # I'm not very happy with this implementation since we call `.eval()`
        # again on terms that are highly likely to be in the model.
        # But it works and it's fine for now.

        if isinstance(encoding, list) and len(encoding) == 1:
            encoding = encoding[0]
        else:
            ValueError("encoding is a list of len > 1")

        evaluated_terms = dict()
        for term in self.terms:
            encoding_ = []
            # encoding is emtpy list when all numerics
            if isinstance(encoding, dict):
                if term.name in encoding.keys():
                    encoding_ = encoding[term.name]
            evaluated_terms[term.name] = term.eval(data, eval_env, encoding_)

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

    def components(self, data, eval_env):
        return {self.name: "Intercept"}

    def eval(self, data, eval_env, encoding):
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

    def components(self, data, eval_env):
        data_cols = data.columns.tolist()
        x = eval_in_data_mask(self.get_eval_str(data_cols), data, eval_env)
        if is_numeric_dtype(x):
            type_ = "numeric"
        elif is_string_dtype(x) or is_categorical_dtype(x) or isinstance(x, dict):
            type_ = "categoric"
        else:
            raise NotImplementedError
        return {self.name: type_}

    def eval(self, data, eval_env, encoding):
        # Workaround: var names present in 'data' are taken from '__DATA__['col']
        # the rest are left as they are and looked up in the upper namespace
        data_cols = data.columns.tolist()
        x = eval_in_data_mask(self.get_eval_str(data_cols), data, eval_env)
        if is_categorical_dtype(x) or is_string_dtype(x):
            return self.eval_categoric(x, encoding)
        elif is_numeric_dtype(x):
            return self.eval_numeric(x)
        else:
            return NotImplemented

    def eval_numeric(self, x):
        if isinstance(x, np.ndarray):
            value = np.atleast_2d(x)
        else:
            value = np.atleast_2d(x.to_numpy()).T
        return {"value": value, "type": "call"}

    def eval_categoric(self, x, encoding):
        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            cat_type = pd.api.types.CategoricalDtype(categories=x.unique().tolist(), ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if isinstance(encoding, list):
            encoding = encoding[0]
        if isinstance(encoding, dict):
            encoding = encoding[self.name]

        if encoding:
            value = pd.get_dummies(x).to_numpy()
            encoding = "full"
        else:
            value = pd.get_dummies(x, drop_first=True).to_numpy()
            encoding = "reduced"
        return {
            "value": value,
            "type": "categoric",
            "levels": levels,
            "reference": reference,
            "encoding": encoding,
        }


class GroupSpecTerm(BaseTerm):
    """Representation of group specific effects term"""

    def __init__(self, expr, factor):
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

    def to_string(self, level=None):
        string = ""
        if isinstance(self.expr, InterceptTerm):
            string += "1|"
        elif isinstance(self.expr, (Term, CallTerm)):
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

    @property
    def vars(self):
        return [self.expr.vars] + [self.factor.vars]

    def eval(self, data, eval_env, encoding):
        # TODO: factor can't be a call or interaction yet.
        if isinstance(self.factor, Term):
            factor = data[self.factor.variable]
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
        Xi = self.expr.eval(data, eval_env, encoding)
        Zi = linalg.khatri_rao(Ji.T, Xi["value"].T).T
        out = {
            "type": Xi["type"],
            "Xi": Xi["value"],
            "Ji": Ji,
            "Zi": sparse.coo_matrix(Zi),
            "groups": factor.cat.categories.tolist(),
        }
        if Xi["type"] == "categoric":
            if "levels" in Xi.keys():
                out["levels"] = Xi["levels"]
                out["reference"] = Xi["reference"]
                out["encoding"] = Xi["encoding"]
            else:
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

    def eval(self, data, eval_env, encoding=None):
        return self.term.eval(data, eval_env, encoding, is_response=True)


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

    def components(self, data, eval_env):
        d = dict()
        for term in self.common_terms:
            if isinstance(term, (InterceptTerm, Term, CallTerm, InteractionTerm)):
                d[term.name] = term.components(data, eval_env)
        return d

    def _encoding_groups(self, data, eval_env):
        """Obtain groups to determine encoding"""

        components = self.components(data, eval_env)

        # First, group with only categoric terms
        categoric_group = dict()
        for k, v in components.items():
            all_categoric = all([t == "categoric" for t in v.values()])
            is_intercept = len(v) == 1 and "Intercept" in v.keys()
            if all_categoric:
                categoric_group[k] = [k_ for k_ in v.keys()]
            if is_intercept:
                categoric_group[k] = []

        # Determine groups of numerics
        numeric_group_sets = []
        numeric_groups = []
        for k, v in components.items():
            categoric = [k_ for k_, v_ in v.items() if v_ == "categoric"]
            numeric = [k_ for k_, v_ in v.items() if v_ == "numeric"]
            if categoric and numeric:
                numeric_set = set(numeric)
                if numeric_set not in numeric_group_sets:
                    numeric_group_sets.append(numeric_set)
                    numeric_groups.append(dict())
                idx = numeric_group_sets.index(numeric_set)
                numeric_groups[idx][k] = categoric

        return [categoric_group] + numeric_groups

    def _encoding_bools(self, data, eval_env):
        """Determine encodings for terms containing at least one categorical variable.

        This method returns dictionaries with True/False values.
        True means the categorical variable uses 'levels' dummies.
        False means the categorial variable uses 'levels - 1' dummies.
        """
        groups = self._encoding_groups(data, eval_env)
        l = [pick_contrasts(group) for group in groups]
        result = dict()
        for d in l:
            result.update(d)
        return result

    def eval(self, data, eval_env):
        encoding = self._encoding_bools(data, eval_env)
        result = dict()
        # Group specific effects aren't evaluated here -- this may change
        for term in self.common_terms:
            term_encoding = None

            if term.name in encoding.keys():
                term_encoding = encoding[term.name]
            else:
                term_encoding = []

            # we're in an interaction that added terms
            # we need to create and evaluate these extra terms
            if len(term_encoding) > 1:
                for term_ in term_encoding:
                    if len(term_) == 1:
                        name = list(term_.keys())[0]
                        encoding = list(term_.values())[0]
                        result[name] = Term(name, name).eval(data, eval_env, encoding)
                    else:
                        # Hack to keep original order, there's somethin happening with sets
                        # in 'contrasts.py'
                        terms_ = term.name.split(":")
                        iterm_ = InteractionTerm(
                            *[term.get_term(name) for name in terms_ if name in term_.keys()]
                        )
                        result[iterm_.name] = iterm_.eval(data, eval_env, term_)
            else:
                result[term.name] = term.eval(data, eval_env, term_encoding)
        return result


ATOMIC_TERMS = (
    Term,
    InteractionTerm,
    CallTerm,
    InterceptTerm,
    NegatedIntercept,
    LiteralTerm,
    GroupSpecTerm,
)
