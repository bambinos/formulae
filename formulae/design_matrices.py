class DesignMatrices:
    """Wraps ResponseVector FixedMatrix and RandomMatrix
    """

    def __init__(self):
        pass


class Base:
    """A base class for ResponseVector, FixedMatrix and RandomMatrix
    """

    def __init__(self):
        pass


class ResponseVector(Base):
    """Representation of the respose vector of a model
    """

    def __init__(self, term):
        self.term = term
        self.y = None
        self.name = None

    def _evaluate(self, data):
        """Evaluates `self.term` inside the data mask provided by `data` and
        updates `self.y` and `self.name`
        """
        pass


class FixedMatrix(Base):
    """Representation of the design matrix for the fixed part of a model.
    """

    def __init__(self, model_terms):
        pass


class RandomMatrix(Base):
    """Representation of the design matrix for the fixed part of a model.
    """

    def __init__(self, model_terms):
        pass