import numpy as np
from .terms import ModelTerms

class DesignMatrices:
    """Wraps ResponseVector CommonEffectsMatrix and GroupEffectsMatrix

    Parameters
    ----------

    model : ModelTerms
        The model description.
    data: DataFrame or dict
        The data object where we take values from.
    """

    def __init__(self, model, data):
        self.model = model

        if self.model.response is not None:
            self.response = ResponseVector(self.model.response, data)
        else:
            self.response = None

        self.common = CommonEffectsMatrix(ModelTerms(*self.model.common_terms), data)
        # self.group = GroupEffectsMatrix(self.model.group_terms, data)


class DesignMatrix:
    """A base class for ResponseVector, CommonEffectsMatrix and GroupEffectsMatrix
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.evaluate()

    def evaluate(self):
        return NotImplemented


class ResponseVector:
    """Representation of the respose vector of a model
    """

    def __init__(self, term, data):
        self.name = None # a string
        self.data = None # 1d numpy array
        self.type = None # either numeric or categorical
        self.refclass = None # Not None for categorical variables
        self.term = term
        self.evaluate(data)

    def evaluate(self, data):
        """Evaluates `self.term` inside the data mask provided by `data` and
        updates `self.y` and `self.name`
        """
        d = self.term.eval(data)
        self.data = d['value']
        self.type = d['type']
        if self.type == 'categoric':
            self.refclass = d.reference


class CommonEffectsMatrix(DesignMatrix):
    """Representation of the design matrix for the common effects of a model.
    """

    def __init__(self, terms, data):
        self.data = None
        self.columns = None # USE SLICES
        self.names = None
        self.terms = terms
        self.evaluate(data)

    def evaluate(self, data):
        d = self.terms.eval(data)
        self.data = np.column_stack([d[key]['value'] for key in d.keys()])

    def _get_sub_matrix(self, column):
        # Method to obtain the columns of the matrix that correspond to a variable.
        pass


class GroupEffectsMatrix(DesignMatrix):
    """Representation of the design matrix for the group specific effects of a model.
    """

    def __init__(self, term, data):
        super().__init__(term, data)

    def evaluate(self):
        pass