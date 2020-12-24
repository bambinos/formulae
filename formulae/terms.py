import itertools

class Term:
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

    def __init__(self, name, variable, data=None, kind=None):
        self.name = name
        self.variable = variable
        self.data = data
        self.kind = kind

    def __hash__(self):
        return hash((self.name, self.variable, self.data, self.kind))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.name == other.name and self.variable == other.variable and self.kind == other.kind

    def __or__(self, other):
        if isinstance(other, (InteractionTerm, NegatedTerm, CallTerm)):
            return ModelTerms(self, other)
        elif isinstance(other, Term):
            if self == other:
                return self
            else:
                return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return other.add_term(self)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            # x-y is equal to x
            return self
        elif isinstance(other, LiteralTerm):
            if other.value in [0, 1]:
                return ModelTerms(self, NegatedTerm("intercept"))
            else:
                # If using other value than 0 or 1, ignore operation
                return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, CallTerm):
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, InteractionTerm):
            return ModelTerms(self, other, other.add_term(self))
        elif isinstance(other, Term):
            print("aaa")
            if self == other:
                return self
            else:
                return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, ModelTerms):
            products = itertools.product({self}, other.terms)
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, CallTerm):
            return InteractionTerm(self, other)
        if isinstance(other, type(self)):
            if self == other:
                return self
            else:
                return InteractionTerm(self, other)
        if isinstance(other, InteractionTerm):
            return other.add_term(self)

    # Maybe I should drop this __pow__
    def __pow__(self, other):
        if isinstance(other, LiteralTerm) and isinstance(other.value, int) and other.value >= 1:
            return self
        else:
            raise ValueError("Bad power")

    def __truediv__(self, other):
        if isinstance(other, (CallTerm, LiteralTerm)):
            return ModelTerms(self, InteractionTerm(self, other))
        elif isinstance(other, InteractionTerm):
            return ModelTerms(self, other.add_term(self))
        if isinstance(other, type(self)):
            if self == other:
                return self
            else:
                return ModelTerms(self, InteractionTerm(self, other))
        else:
            raise ValueError("Bad truediv")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "name= " + self.name,
            "variable= " + self.variable,
            "kind= " + str(self.kind),
            "data= " + str(self.data)
        ]
        return 'Term(\n  ' + '\n  '.join(string_list) + '\n)'

class InteractionTerm:
    """Representation of an interaction term

    Parameters
    ----------
    name : str
        Name of the term.
    terms: list
        list of Terms taking place in the interaction
    """

    def __init__(self, *terms):
        # self.terms is a list because I have to admit repeated terms
        # self.variables is a set because i want to store each variable once
        # but there must be a better way to do this
        self.variables = set()
        self.terms = []
        for term in terms:
            self.add_term(term)

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.name == other.name and self.terms == other.terms and self.variables == other.variables

    def add_term(self, term):
        if isinstance(term, Term):
            self.terms.append(term)
            self.variables.add(term.variable)
            return self
        elif isinstance(term, CallTerm):
            self.terms.append(term)
            self.variables.add(term.call)
        elif isinstance(term, InteractionTerm):
            self.terms = self.terms + term.terms
            self.variables.update(term.variables)
            return self
        else:
            return NotImplemented

    @property
    def name(self):
        return ":".join([term.name for term in self.terms])

    def __or__ (self, other):
        return ModelTerms(self, other)

    def __matmul__(self, other):
        if isinstance(other, (type(self), Term, CallTerm)):
            return self.add_term(other)
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

class LiteralTerm:
    def __init__(self, value):
        self.value = value
        self.name = str(self.value)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"LiteralTerm(value={self.value})"

class NegatedTerm:
    def __init__(self, what):
        self.what = what

    def __hash__(self):
        return hash((self.what))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.what == other.what

    def __or__(self, other):
        if isinstance(other, Term):
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return other.add_term(self)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"NegatedTerm(\n  what={self.what}\n)"


class CallTerm:
    """Representation of a call term
    """

    def __init__(self, expr):
        self.call = expr.callee.name.lexeme + expr.arguments.lexeme
        self.name = self.call
        self.special = expr.special

    def __hash__(self):
        return hash((self.call, self.special))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.call == other.call and self.special == other.special

    def __or__(self, other):
        if isinstance(other, (Term, InteractionTerm, CallTerm, NegatedTerm, LiteralTerm)):
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return other.add_term(self)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (Term, InteractionTerm, CallTerm, NegatedTerm)):
            # x-y is equal to x
            return self
        elif isinstance(other, LiteralTerm):
            if other.value in [0, 1]:
                return ModelTerms(self, NegatedTerm("intercept"))
            else:
                # If using other value than 0 or 1, ignore operation
                return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (Term, InteractionTerm, CallTerm, LiteralTerm)):
            return ModelTerms(self, other, InteractionTerm(self, other))
        elif isinstance(other, ModelTerms):
            products = itertools.product({self}, other.terms)
            terms = [self] + list(other.terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) | ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, (Term, CallTerm, LiteralTerm)):
            return InteractionTerm(self, other)
        elif isinstance(other, InteractionTerm):
            return other.add_term(self)
        elif isinstance(other, ModelTerms):
            products = itertools.product({self}, other.terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) | ModelTerms(*iterms)
        else:
            return NotImplemented

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        strlist = [
            "call=" + self.call,
            "special=" + str(self.special)
        ]
        return 'CallTerm(\n  ' + ',\n  '.join(strlist) + '\n)'

class ResponseTerm:
    """Representation of a response term
    """

    # TODO: things like {x - y} must be a Term and not ModelTerms
    def __init__(self, term):
        if isinstance(term, Term):
            self.term = term
            self.name = term.name
            self.variable = term.variable
            self.kind = term.kind
            self.data = term.data
        else:
            raise ValueError("Response Term must be univariate")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "name= " + self.name,
            "variable= " + self.variable,
            "kind= " + str(self.kind),
            "data= " + str(self.data)
        ]
        return 'ResponseTerm(\n  ' + '\n  '.join(string_list) + '\n)'

    def __or__(self, other):
        if isinstance(other, (Term, InteractionTerm)):
            return ModelTerms(other, response=self)
        elif isinstance(other, ModelTerms):
            return other.add_response(self)
        else:
            return NotImplemented

class ModelTerms:
    accepted_terms = (Term, InteractionTerm, NegatedTerm, CallTerm)

    def __init__(self, *terms, response=None):

        if isinstance(response, ResponseTerm) or response is None:
            self.response = response
        else:
            raise ValueError("bad ResponseTerm")

        if all([isinstance(term, self.accepted_terms) for term in terms]):
            self.terms = set([term for term in terms])
        else:
            raise ValueError("All terms must be of class Term, InteractionTerm or NegatedTerm")

    def add_response(self, term):
        if isinstance(term, ResponseTerm):
            self.response = term
            return self
        else:
            raise ValueError("not ResponseTerm")

    def add_term(self, term):
        if isinstance(term, self.accepted_terms):
            self.terms.add(term)
            return self
        else:
            raise ValueError("not accepted term")

    def __sub__(self, other):
        if isinstance(other, type(self)):
            self.terms = self.terms - other.terms
            return self
        elif isinstance(other, Term):
            self.terms = self.terms - {other}
            return self
        else:
            return NotImplemented

    def __or__(self, other):
        if isinstance(other, self.accepted_terms):
            self.add_term(other)
            return self
        elif isinstance(other, type(self)):
            self.terms = self.terms | set([term for term in other.terms])
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, type(self)):
            products = itertools.product(self.terms, other.terms)
            terms = list(self.terms) + list(other.terms)
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) | ModelTerms(*iterms)
        elif isinstance(other, Term):
            products = itertools.product(self.terms, {other})
            terms = [term for term in self.terms] + [other]
            iterms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms) | ModelTerms(*iterms)
        else:
            return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            products = itertools.product(self.terms, other.terms)
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        elif isinstance(other, Term):
            products = itertools.product(self.terms, {other})
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        else:
            return NotImplemented

    def __repr__(self):
        terms = ',\n'.join([repr(term) for term in self.terms])
        if self.response is None:
            string = '  '.join(terms.splitlines(True))
        else:
            string = '  '.join(str(self.response).splitlines(True))
            string += ',\n  ' + '  '.join(terms.splitlines(True))

        return 'ModelTerms(\n  ' + string + '\n)'

