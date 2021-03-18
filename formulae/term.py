from itertools import product

class TermComponent:
    """Atomic component of a Term"""

    def __init__(self, name):
        self.name = name


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

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.components == other.components

    def __add__(self, other):
        if isinstance(other, type(self)):
            if self == other:
                return self
            else:
                return ModelTerms(self, other)
        elif isinstance(other, ModelTerms):
            return ModelTerms(self) + other
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (type(self), ModelTerms)):
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, type(self)):
            if self == other:
                return self
            else:
                return ModelTerms(self, other, Term(*self.components, *other.components))
        elif isinstance(other, ModelTerms):
            products = product([self], other.common_terms)
            terms = [self] + other.common_terms
            iterms = [Term(*p[0].components, *p[1].components) for p in products]
            return ModelTerms(*terms) + ModelTerms(*iterms)
        else:
            return NotImplemented

class ModelTerms:
    """Representation of the terms in a model"""

    def __init__(self):
        pass
