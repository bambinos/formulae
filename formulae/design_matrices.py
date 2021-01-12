class DesignMatrices:
    """Wraps ResponseVector FixedMatrix and RandomMatrix

    Parameters
    ----------

    model : ModelTerms
        The model description.
    data: DataFrame or dict
        The data object where we take values from.
    """

    def __init__(self, model, data):
        self.model = model
        self.response = ResponseVector(self.model.response, data)
        self.fixed = FixedMatrix(self.model.terms, data)
        self.random = RandomMatrix(self.model.random_terms, data)


class Base:
    """A base class for ResponseVector, FixedMatrix and RandomMatrix
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.evaluate()

    def evaluate(self):
        return NotImplemented


class ResponseVector(Base):
    """Representation of the respose vector of a model
    """

    def __init__(self, term, data):
        super().__init__(term, data)

    def evaluate(self):
        """Evaluates `self.term` inside the data mask provided by `data` and
        updates `self.y` and `self.name`
        """
        pass


class FixedMatrix(Base):
    """Representation of the design matrix for the fixed part of a model.
    """

    def __init__(self, term, data):
        super().__init__(term, data)

    def evaluate(self):
        pass


class RandomMatrix(Base):
    """Representation of the design matrix for the fixed part of a model.
    """

    def __init__(self, term, data):
        super().__init__(term, data)

    def evaluate(self):
        pass