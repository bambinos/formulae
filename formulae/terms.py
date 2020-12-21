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
        if not isinstance(other, type(self)): 
            return NotImplemented
            
        if self == other:
            return ModelTerms(self)
        else:
            return ModelTerms(self, other)
    
    def __sub__(self, other):
        if not isinstance(other, type(self)): 
            return NotImplemented
        # "x" - "y" is equal to "x"
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


class ModelTerms:
    # TODO: Add accept method, so we accept and unpack ModelTerms
    accepted_terms = (Term, InteractionTerm)
    def __init__(self, *terms):
        if all([isinstance(term, self.accepted_terms) for term in terms]):
            self.terms = set([term for term in terms])
        else:
            raise ValueError("not term")
           
    def __sub__(self, other):
        if isinstance(other, type(self)):
            self.terms = self.terms - other.terms
            return self
        elif isinstance(other, Term):
            self.terms = self.terms - {other}
            return self
        else:
            return NotImplemented
            
    
    def __repr__(self):
        terms = ',\n'.join([repr(term) for term in self.terms])
        return 'ModelTerms(\n  ' + '  '.join(terms.splitlines(True)) + '\n)'
            
           
            

