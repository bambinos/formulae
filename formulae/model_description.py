from formulae.terms.terms import Model

from formulae.scanner import Scanner
from formulae.parser import Parser
from formulae.resolver import Resolver


def model_description(formula):
    """Interpret model formula and obtain a model description.

    This function receives a string with a formula describing a statistical
    model and returns an object of class ModelTerms that describes the
    model interpreted from the formula.

    Parameters
    ----------
    formula: string
        A string with a model description in formula language.

    Returns
    ----------
    An object of class ModelTerms with an internal description of the model.
    """

    description = Resolver(Parser(Scanner(formula).scan()).parse()).resolve()

    if isinstance(description, Model):
        return description

    return Model(description)
