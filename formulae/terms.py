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
    fun: callable
        An optional function used to transform the values of the term.
        Only applies for 'numeric' variables for now.
    fargs: dictionary
        An optional dictionary where keys are argument names and values are argument values
        that are passed to the callable 'fun'.
    """

    def __init__(self, name, variables, kind=None, fun=None, fargs=None):
        self.name = name
        self.variables = variables
        self.kind = kind
        self.fun = fun
        self.fargs = fargs

