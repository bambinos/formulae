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
        if isinstance(other, Term):
            if self == other:
                return self
            else:
                return ModelTerms(self, other)
        elif isinstance(other, InteractionTerm):
            return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return other.add_term(self)
        else:
            return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        # x-y is equal to x
        return self

    def __mul__(self, other):
        if isinstance(other, Term):
            if self == other:
                return self
            else:
                # name = f"{self.name}:{other.name}"
                return ModelTerms(self, other, InteractionTerm(self, other))
        if isinstance(other, ModelTerms):
            products = itertools.product({self}, other.terms)
            #terms = [InteractionTerm(f"{p[0].name}:{p[1].name}", p[0], p[1]) for p in products]
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            if self == other:
                return self
            else:
                # name = f"{self.name}:{other.name}"
                return InteractionTerm(self, other)
        if isinstance(other, InteractionTerm):
            return other.add_term(self)

    def __pow__(self, other):
        if isinstance(other, LiteralTerm) and isinstance(other.value, int) and other.value >= 1:
            return self
            # if other.value == 1:
            #    return self
            # else:
            #    terms = [self for i in range(other.value)]
            #    name = ":".join([self.name for i in range(other.value)])
            #    return InteractionTerm(name, *terms)
        else:
            raise ValueError("Bad power")

    def __truediv__(self, other):
        if isinstance(other, Term):
            # name = f"{self.name}:{other.name}"
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
        #self.name = name
        self.variables = set()
        self.terms = []
        for term in terms:
            self.add_term(term)
        #self.terms = [term for term in terms]
        #self.variables = {term.variable for term in terms}

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.name == other.name and self.terms == other.terms and self.variables == other.variables

    def add_term(self, term):
        if isinstance(term, Term):
            self.terms.append(term)
            self.variables.add(term.variable)
            # self.name += f":{term.name}"
            return self
        elif isinstance(term, InteractionTerm):
            self.terms = self.terms + term.terms
            self.variables.update(term.variables)
            # self.name += f":{term.name}"
            return self
        else:
            return NotImplemented

    @property
    def name(self):
        return ":".join([term.name for term in self.terms])

    def __or__ (self, other):
        return ModelTerms(self, other)

    def __matmul__(self, other):
        if isinstance(other, (type(self), Term)):
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

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"LiteralTerm(value={self.value})"

class NegatedTerm:
    def __init__(self, what):
        self.what = what

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"NegatedTerm(what={self.what})"


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
    # TODO: Add accept method, so we accept and unpack ModelTerms
    accepted_terms = (Term, InteractionTerm)

    def __init__(self, *terms, response=None):

        if isinstance(response, ResponseTerm) or response is None:
            self.response = response
        else:
            raise ValueError("bad ResponseTerm")

        if all([isinstance(term, self.accepted_terms) for term in terms]):
            self.terms = set([term for term in terms])
        else:
            raise ValueError("not term")

    def add_response(self, term):
        if isinstance(term, ResponseTerm):
            self.response = term
            return self
        else:
            raise ValueError("not ResponseTerm")

    def add_term(self, term):
        if isinstance(term, self.accepted_terms):
            self.terms.add(term)
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

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            products = itertools.product(self.terms, other.terms)
            # terms = [InteractionTerm(f"{p[0].name}:{p[1].name}", p[0], p[1]) for p in products]
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        elif isinstance(other, Term):
            products = itertools.product(self.terms, {other})
            #terms = [InteractionTerm(f"{p[0].name}:{p[1].name}", p[0], p[1]) for p in products]
            terms = [InteractionTerm(p[0], p[1]) for p in products]
            return ModelTerms(*terms)
        #elif isinstance(other, InteractionTerm):
        #    terms = [other.add_term(term) for term in self.terms]
        #    return ModelTerms(*terms)
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

