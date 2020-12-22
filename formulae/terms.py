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
        if not isinstance(other, (Term, ModelTerms)):
            return NotImplemented

        if isinstance(other, Term):
            if self == other:
                return self
            else:
                return ModelTerms(self, other)
        else:
            return other.add_term(self)

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        # x-y is equal to x
        return self


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

    def __init__(self, name, terms):
        self.name = name
        self.terms = self.assert_terms(terms)

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

    def __repr__(self):
        terms = ',\n'.join([repr(term) for term in self.terms])
        if self.response is None:
            string = '  '.join(terms.splitlines(True))
        else:
            string = '  '.join(str(self.response).splitlines(True))
            string += ',\n  ' + '  '.join(terms.splitlines(True))

        return 'ModelTerms(\n  ' + string + '\n)'

