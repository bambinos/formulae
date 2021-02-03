from .scanner import Scanner
from .parser import Parser
from .resolver import Resolver


def model_description(formula):
    return Resolver(Parser(Scanner(formula).scan()).parse()).resolve()