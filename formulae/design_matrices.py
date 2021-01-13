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
        self.fixed = FixedEffectsMatrix(self.model.terms, data)
        self.random = RandomEffectsMatrix(self.model.random_terms, data)


class DesignMatrix:
    """A base class for ResponseVector, FixedEffectsMatrix and RandomEffectsMatrix
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.evaluate()

    def evaluate(self):
        return NotImplemented


class ResponseVector(DesignMatrix):
    """Representation of the respose vector of a model
    """

    def __init__(self, term, data):
        super().__init__(term, data)

    def evaluate(self):
        """Evaluates `self.term` inside the data mask provided by `data` and
        updates `self.y` and `self.name`
        """
        pass


class FixedEffectsMatrix(DesignMatrix):
    """Representation of the design matrix for the fixed part of a model.
    """

    def __init__(self, term, data):
        super().__init__(term, data)

    def evaluate(self):
        pass


class RandomEffectsMatrix(DesignMatrix):
    """Representation of the design matrix for the fixed part of a model.
    """

    def __init__(self, term, data):
        super().__init__(term, data)

    def evaluate(self):
        pass