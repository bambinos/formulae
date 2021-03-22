import numpy as np
import pandas as pd

from itertools import combinations, product
from pandas.api.types import is_categorical_dtype, is_numeric_dtype, is_string_dtype

from .call_utils import CallEvalPrinter, CallNamePrinter, CallVarsExtractor
from .eval import eval_in_data_mask
from .transforms import STATEFUL_TRANSFORMS

class Variable:
    """Atomic component of a Term"""

    def __init__(self, name, level=None):
        self.data = None
        self.type_ = None
        self.name = name
        self.level = level

    def __hash__(self):
        return hash((self.type_, self.name, self.level))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.type_ == other.type_ and self.name == other.name and self.level == other.level

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, level='{self.level}')"

    @property
    def vars(self):
        return self.name

    def set_type(self, data_mask):
        # TODO: Drop `eval_env` argument
        """Detemines the type of the variable.

        Looks for the name of the variable in ``data`` and sets the ``.type_`` property to
        ``"numeric"`` or ``"categoric"`` depending on the type of the variable.
        """
        x = data_mask[self.name]
        if is_numeric_dtype(x):
            self.type_ = "numeric"
        elif is_string_dtype(x) or is_categorical_dtype(x):
            self.type_ = "categoric"
        else:
            raise ValueError(f"Variable is of an unrecognized type ({type(x)}).")

    def set_data(self, data_mask, encoding, is_response=False):
        """Obtains and stores the final data object related to this variable.

        Evaluates the variable according to its type and stores the result in ``.data_mask``. It
        does not support multi-level categoric responses yet. If ``is_response`` is ``True`` and the
        variable is of a categoric type, this method returns a 1d array of 0-1 instead of a matrix.
        """

        if self.type_ is None:
            raise ValueError("Variable type is not set.")
        if self.type_ not in ["numeric", "categoric"]:
            raise ValueError(f"Variable is of an unrecognized type ({self.type_}).")
        x = data_mask[self.name]
        if self.type_ == "numeric":
            self.data = self.eval_numeric(x)
        elif self.type_ == "categoric":
            self.data = self.eval_categoric(x, encoding, is_response)
        else:
            raise ValueError("Unexpected error while trying to evaluate a Variable.")

    def eval_numeric(self, x):
        if self.level is not None:
            raise ValueError("Subset notation can't be used with a numeric variable.")
        out = {"value": np.atleast_2d(x.to_numpy()).T, "type": "numeric"}
        return out

    def eval_categoric(self, x, encoding, is_response):
        # If not ordered, we make it ordered
        # x.unique() preservese order of appearence
        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            categories = sorted(x.unique().tolist())
            cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
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


class Call:
    """Atomic component of a Term that is a call"""

    def __init__(self, expr):
        self.data = None
        self._raw_data = None
        self.type_ = None
        self.callee = expr.callee.name.lexeme
        self.args = expr.args
        self.name = self._name_str()
        if self.callee in STATEFUL_TRANSFORMS.keys():
            self.stateful_transform = STATEFUL_TRANSFORMS[self.callee]

    def __hash__(self):
        return hash((self.callee, self.args, self.name, self.stateful_transform))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.callee == other.callee
            and self.name == other.name
            and self.args == other.args
            and self.stateful_transform == other.stateful_transform
            )

    def accept(self, visitor):
        return visitor.visitCallTerm(self)

    def _eval_str(self, data_cols):
        return CallEvalPrinter(self, data_cols).print()

    def _name_str(self):
        return CallNamePrinter(self).print()

    @property
    def vars(self):
        return CallVarsExtractor(self).get()

    def set_type(self, data_mask, eval_env):
        """Detemines the type of the result of the call.

        Evaluates the function call and sets the ``.type_`` property to ``"numeric"`` or
        ``"categoric"`` depending on the type of the result. It also stores the intermediate result
        of the evaluation in ``._raw_data`` to prevent us from computing the same thing more than
        once.
        """
        names = data_mask.columns.tolist()
        if self.stateful_transform is not None:
            # Q: How to set non data dependent parameters?
            self.stateful_transform.set_params()
            x = self.stateful_transform.call()
        else:
            x = eval_in_data_mask(self._eval_str(names), data_mask, eval_env)
        if is_numeric_dtype(x):
            self.type_ = "numeric"
            self._raw_data = x
        elif is_string_dtype(x) or is_categorical_dtype(x) or isinstance(x, dict):
            self.type_ = "categoric"
            self._raw_data = x
        else:
            raise ValueError(f"Call result is of an unrecognized type ({type(x)}).")

    def set_data(self, encoding, is_response=False):
        """Obtains and stores the final data object related to this call.

        Evaluates the call according to its type and stores the result in ``.data``. It does not
        support multi-level categoric responses yet. If ``is_response`` is ``True`` and the variable
        is of a categoric type, this method returns a 1d array of 0-1 instead of a matrix.

        In practice it completes the evaluation that started with ``self.set_type()``.
        """
        # Workaround: var names present in 'data' are taken from '__DATA__['col']
        # the rest are left as they are and looked up in the upper namespace
        if self.type_ is None:
            raise ValueError("Call result type is not set.")
        if self.type_ not in ["numeric", "categoric"]:
            raise ValueError(f"Call result is of an unrecognized type ({self.type_}).")
        if self.type_ == "numeric":
            self.data = self.eval_numeric(self._raw_data)
        else:
            self.data = self.eval_categoric(self._raw_data, encoding, is_response)

    def eval_numeric(self, x):
        if isinstance(x, np.ndarray):
            value = np.atleast_2d(x)
        else:
            value = np.atleast_2d(x.to_numpy()).T
        return {"value": value, "type": "numeric"}

    def eval_categoric(self, x, encoding, is_response):
        if not hasattr(x.dtype, "ordered") or not x.dtype.ordered:
            categories = sorted(x.unique().tolist())
            cat_type = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
            x = x.astype(cat_type)

        reference = x.min()
        levels = x.cat.categories.tolist()

        if is_response:
            if self.level is not None:
                reference = self.level
            value = np.atleast_2d(np.where(x == reference, 1, 0)).T
            encoding = None
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

class Term:
    """Representation of a single term in a ModelTerms.

    A model term can be an intercept, a single component, a function call, an interaction
    involving components and/or function calls or a group specific term.
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
        if self == other:
            return self
        elif isinstance(other, type(self)):
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
        """Power of a Term.

        It leaves the term as it is. For a mathematical power, do `I(x ** n)` or `{x ** n}`.
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
        """Group specific term

        (x|g) -> (1|g) + (x|g)
        (x|g + h) -> (x|g) + (x|h) -> (1|g) + (1|h) + (x|g) + (x|h)
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


class Intercept:
    def __init__(self):
        self.name = "Intercept"

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

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

    def components(self, data, eval_env):
        return {self.name: "Intercept"}

class NegatedIntercept:
    def __init__(self):
        self.name = "Intercept"

    def __add__(self, other):
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
        if not isinstance(other, type(self)):
            return False
        else:
            return self.name == other.name

    def __or__(self, other):
        raise ValueError("At least include an intercept in '|' operation")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__}()"

    @property
    def vars(self):
        return ""


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

class ModelTerms:
    """Representation of the terms in a model"""

    def __init__(self, *terms, response=None):
        if isinstance(response, ResponseTerm) or response is None:
            self.response = response
        else:
            raise ValueError("Response must be of class ResponseTerm.")
        ACCEPTED_TERMS = (Term, GroupSpecTerm, Intercept, NegatedIntercept)
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