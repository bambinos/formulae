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
        self.response = None
        self.common = None
        self.group = None
        self.model = model

        if self.model.response is not None:
            self.response = ResponseVector(self.model.response, data)

        if self.model.common_terms is not None:
            self.common = CommonEffectsMatrix(ModelTerms(*self.model.common_terms), data)

        if self.model.group_terms is not None:
            self.group = GroupEffectsMatrix(self.model.group_terms, data)

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
            self.refclass = d['reference']

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            'name=' + self.term.term.name,
            'type=' + self.type,
            "length=" + str(len(self.data))
        ]
        if self.type == 'categoric':
            string_list += ['refclass=' + self.refclass]
        return 'ResponseVector(' + ', '.join(string_list) + ')'

class CommonEffectsMatrix:
    """Representation of the design matrix for the common effects of a model.
    """

    def __init__(self, terms, data):
        self.data = None
        self._terms_info = None
        self.terms = terms
        self.evaluate(data)

    def evaluate(self, data):
        d = self.terms.eval(data)
        self.data = np.column_stack([d[key]['value'] for key in d.keys()])
        self.terms_info = {}
        # Get types and column slices
        start = 0
        for key in d.keys():
            delta = d[key]['value'].shape[1]
            self.terms_info[key] = {
                'type': d[key]['type'],
                'cols': slice(start, start + delta)
            }
            start += delta

    def __getitem__(self, term):
        if term not in self.terms_info.keys():
            raise ValueError(f"'{term}' is not a valid term name")
        else:
            return self.data[:, self.terms_info[term]['cols']]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        terms_list = []
        for key, value in self.terms_info.items():
            terms_list.append(f"'{key}': {{type={value['type']}, cols={str(value['cols'])}}}")
        terms_str = ',\n  '.join(terms_list)
        string_list = [
            'shape=' + str(self.data.shape),
            'terms={\n    ' + '  '.join(terms_str.splitlines(True))+ '\n  }'
        ]
        return 'CommonEffectsMatrix(\n  ' + ',\n  '.join(string_list) + '\n)'


class GroupEffectsMatrix:
    """Representation of the design matrix for the group specific effects of a model.
    """

    def __init__(self, term, data):
        pass

    def evaluate(self):
        pass