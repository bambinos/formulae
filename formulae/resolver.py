from .terms import Term, InteractionTerm, CallTerm, LiteralTerm, ResponseTerm, NegatedTerm
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
            # right must be an integer
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
            if isinstance(expr.right, Literal) and expr.right.value in [0, 1]:
                return NegatedTerm("intercept")
            else:
                # Maybe return something like EmptyTerm so something like 'y ~ -x'
                # works as if it only had intercept?
                raise ResolverError("Unary negation can only be applied to '0' or '1'")
        else:
            raise ResolverError("Couldn't resolve UnaryExpr with otype '" + otype + "'")

    def visitCallExpr(self, expr):
        return CallTerm(expr)

    def visitVariableExpr(self, expr):
        return Term(expr.name.lexeme, expr.name.lexeme)

    def visitLiteralExpr(self, expr):
        return LiteralTerm(expr.value)
