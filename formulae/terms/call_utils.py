from formulae.utils import flatten_list


class CallVarsExtractor:
    """Visitor that extracts variable names present in a model formula"""

    def __init__(self, expr):
        self.expr = expr

    def get(self):
        return list(flatten_list(self.expr.accept(self)))

    def visitCallTerm(self, term):
        return term.call.accept(self)

    def visitAssignExpr(self, expr):
        return str(expr.value.accept(self))

    def visitGroupingExpr(self, expr):
        return expr.expression.accept(self)

    def visitBinaryExpr(self, expr):
        return [expr.left.accept(self), expr.right.accept(self)]

    def visitUnaryExpr(self, expr):
        return expr.right.accept(self)

    def visitCallExpr(self, expr):
        return list(flatten_list([arg.accept(self) for arg in expr.args]))

    def visitVariableExpr(self, expr):
        return expr.name.lexeme

    def visitLiteralExpr(self, expr):  # pylint: disable = unused-argument
        return ""

    def visitQuotedNameExpr(self, expr):
        # delete backquotes in 'variable'
        return expr.expression.lexeme[1:-1]

    def visitLazyOperator(self, expr):
        return list(arg.accept(self) for arg in expr.args)

    def visitLazyVariable(self, expr):
        return expr.name

    def visitLazyValue(self, expr):  # pylint: disable = unused-argument
        return ""

    def visitLazyCall(self, expr):
        args = list(flatten_list([arg.accept(self) for arg in expr.args]))
        kwargs = list(flatten_list([arg.accept(self) for arg in expr.kwargs.values()]))
        return args + kwargs
