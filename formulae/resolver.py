from .terms import Term, InteractionTerm, CallTerm, LiteralTerm, ResponseTerm, InterceptTerm, NegatedIntercept
from .expr import Literal

class ResolverError(Exception):
    pass

class Resolver:

    def __init__(self, expr):
        self.expr = expr

    def resolve(self):
        return self.expr.accept(self)

    def visitGroupingExpr(self, expr):
        return expr.expression.accept(self)

    def visitBinaryExpr(self, expr):
        otype = expr.operator.type
        if otype == 'TILDE':
            return ResponseTerm(expr.left.accept(self)) + expr.right.accept(self)
        if otype == 'PLUS':
            return expr.left.accept(self) + expr.right.accept(self)
        elif otype == 'MINUS':
            return expr.left.accept(self) - expr.right.accept(self)
        elif otype == 'STAR_STAR':
            return expr.left.accept(self) ** expr.right.accept(self)
        elif otype == 'COLON':
            # there is not __colon__ method
            return expr.left.accept(self) @ expr.right.accept(self)
        elif otype == 'STAR':
            return expr.left.accept(self) * expr.right.accept(self)
        elif otype == 'SLASH':
            return expr.left.accept(self) / expr.right.accept(self)
        elif otype == 'PIPE':
            return expr.left.accept(self) | expr.right.accept(self)
        else:
            raise ResolverError("Couldn't resolve BinaryExpr with otype '" + otype + "'")

    def visitUnaryExpr(self, expr):
        otype = expr.operator.type
        if otype == 'PLUS':
            return expr.right.accept(self)
        elif otype == 'MINUS':
            expr = expr.right.accept(self)
            if isinstance(expr, InterceptTerm):
                return NegatedIntercept()
            elif isinstance(expr, NegatedIntercept):
                return InterceptTerm()
            else:
                raise ResolverError("Unary negation can only be applied to '0' or '1'")
        else:
            raise ResolverError("Couldn't resolve UnaryExpr with otype '" + otype + "'")

    def visitCallExpr(self, expr):
        return CallTerm(expr)

    def visitVariableExpr(self, expr):
        return Term(expr.name.lexeme, expr.name.lexeme)

    def visitLiteralExpr(self, expr):
        if expr.value == 0:
            return NegatedIntercept()
        elif expr.value == 1:
            return InterceptTerm()
        else:
            return LiteralTerm(expr.value)
