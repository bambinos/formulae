from .terms import (
    Variable,
    Call,
    Term,
    Intercept,
    NegatedIntercept,
    ResponseTerm
)


class ResolverError(Exception):
    pass


class Resolver:
    """Visitor that walks through the AST and returns a model description"""

    def __init__(self, expr):
        self.expr = expr

    def resolve(self):
        return self.expr.accept(self)

    def visitGroupingExpr(self, expr):
        return expr.expression.accept(self)

    def visitBinaryExpr(self, expr):
        otype = expr.operator.type
        if otype == "TILDE":
            return ResponseTerm(expr.left.accept(self)) + expr.right.accept(self)
        if otype == "PLUS":
            return expr.left.accept(self) + expr.right.accept(self)
        elif otype == "MINUS":
            return expr.left.accept(self) - expr.right.accept(self)
        elif otype == "STAR_STAR":
            return expr.left.accept(self) ** expr.right.accept(self)
        elif otype == "COLON":
            # there is not __colon__ method
            return expr.left.accept(self) @ expr.right.accept(self)
        elif otype == "STAR":
            return expr.left.accept(self) * expr.right.accept(self)
        elif otype == "SLASH":
            return expr.left.accept(self) / expr.right.accept(self)
        elif otype == "PIPE":
            return expr.left.accept(self) | expr.right.accept(self)
        else:
            raise ResolverError("Couldn't resolve BinaryExpr with otype '" + otype + "'")

    def visitUnaryExpr(self, expr):
        otype = expr.operator.type
        if otype == "PLUS":
            return expr.right.accept(self)
        elif otype == "MINUS":
            expr = expr.right.accept(self)
            if isinstance(expr, Intercept):
                return NegatedIntercept()
            elif isinstance(expr, NegatedIntercept):
                return Intercept()
            else:
                raise ResolverError("Unary negation can only be applied to '0' or '1'")
        else:
            raise ResolverError("Couldn't resolve UnaryExpr with otype '" + otype + "'")

    def visitCallExpr(self, expr):
        return Term(Call(expr))

    def visitVariableExpr(self, expr):
        return Term(Variable(expr.name.lexeme, expr.level))

    def visitLiteralExpr(self, expr):
        if expr.value == 0:
            return NegatedIntercept()
        elif expr.value == 1:
            return Intercept()
        else:
            return Term(Variable(expr.value))

    def visitQuotedNameExpr(self, expr):
        # delete backquotes in 'variable' and 'name'
        return Term(expr.expression.lexeme[1:-1], expr.expression.lexeme[1:-1])


# When evaluating ModelTerms object we'll have to "evaluate" CallTerms in a different manner
# because the arguments is expressed as an AST and not as a ModelTerms object.