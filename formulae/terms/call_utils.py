import pandas as pd
from formulae.utils import flatten_list


class CallEvalPrinter:
    """Visitor that generates function calls code to be evaluated in Python."""

    def __init__(self, expr, data_names=None):
        self.expr = expr
        self.data_names = data_names

    def print(self):
        return self.expr.accept(self)

    def visitCallTerm(self, term):
        args = ", ".join([str(arg.accept(self)) for arg in term.args])
        return term.callee + "(" + args + ")"

    def visitAssignExpr(self, expr):
        return expr.name.name.lexeme + " = " + str(expr.value.accept(self))

    def visitGroupingExpr(self, expr):
        return expr.expression.accept(self)

    def visitBinaryExpr(self, expr):
        return (
            str(expr.left.accept(self))
            + " "
            + str(expr.operator.lexeme)
            + " "
            + str(expr.right.accept(self))
        )

    def visitUnaryExpr(self, expr):
        return str(expr.operator.lexeme) + " " + str(expr.right.accept(self))

    def visitCallExpr(self, expr):
        args = ", ".join([arg.accept(self) for arg in expr.args])
        return expr.callee.name.lexeme + "(" + args + ")"

    def visitVariableExpr(self, expr):
        name = expr.name.lexeme
        if name in self.data_names:
            return "__DATA__['" + name + "']"
        else:
            return name

    def visitLiteralExpr(self, expr):
        return expr.value

    def visitQuotedNameExpr(self, expr):
        # delete backquotes in 'variable'
        return "__DATA__['" + expr.expression.lexeme[1:-1] + "']"


class CallNamePrinter(CallEvalPrinter):
    def visitVariableExpr(self, expr):
        return expr.name.lexeme

    def visitQuotedNameExpr(self, expr):
        return expr.expression.lexeme


class CallVarsExtractor:
    """Visitor that extracts variable names present in a model formula"""

    def __init__(self, expr):
        self.expr = expr

    def get(self):
        x = self.expr.accept(self)
        # make sure it return a list
        return list(flatten_list(x))

    def visitCallTerm(self, term):
        return [arg.accept(self) for arg in term.args]

    def visitAssignExpr(self, expr):
        return str(expr.value.accept(self))

    def visitGroupingExpr(self, expr):
        return expr.expression.accept(self)

    def visitBinaryExpr(self, expr):
        return [expr.left.accept(self), expr.right.accept(self)]

    def visitUnaryExpr(self, expr):
        return expr.right.accept(self)

    def visitCallExpr(self, expr):
        return [arg.accept(self) for arg in expr.args]

    def visitVariableExpr(self, expr):
        return expr.name.lexeme

    def visitLiteralExpr(self, expr):  # pylint: disable = unused-argument
        return ""

    def visitQuotedNameExpr(self, expr):
        # delete backquotes in 'variable'
        return expr.expression.lexeme[1:-1]


def get_data_mask_names(data_mask):
    if isinstance(data_mask, pd.DataFrame):
        return data_mask.columns.tolist()
    elif isinstance(data_mask, dict):
        return list(dict.keys())
    else:
        raise ValueError("'data_mask' must be a pandas DataFrame or a dictionary.")
