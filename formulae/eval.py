class Evaluator:
    """Resolves objects of class BaseTerm and similar
    """

    def __init__(self, terms):
        self.terms = terms

    def eval(self, data):
        return self.terms.eval(data)