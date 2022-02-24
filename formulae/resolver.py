from formulae.terms import Variable, Call, Term, Intercept, NegatedIntercept, Response
from formulae.terms.call_resolver import CallResolver


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

    def visitBinaryExpr(self, expr):  # pylint: disable=too-many-return-statements
        otype = expr.operator.kind
        if otype == "TILDE":
            return Response(expr.left.accept(self)) + expr.right.accept(self)
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
        else:  # pragma: no cover
            raise ResolverError("Couldn't resolve BinaryExpr with otype '" + otype + "'")

    def visitUnaryExpr(self, expr):
        otype = expr.operator.kind
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
        else:  # pragma: no cover
            raise ResolverError("Couldn't resolve UnaryExpr with otype '" + otype + "'")

    def visitCallExpr(self, expr):
        # Delegates all the work to self._visitCallExpr, that works recursively through its args.
        # It just wraps the result in a Term object.
        return Term(Call(CallResolver(expr).resolve()))

    def visitVariableExpr(self, expr):
        if expr.level:
            level = expr.level.value
        else:
            level = None
        return Term(Variable(expr.name.lexeme, level))

    def visitLiteralExpr(self, expr):
        if expr.value == 0:
            return NegatedIntercept()
        elif expr.value == 1:
            return Intercept()
        else:
            return Term(Variable(expr.value))

    def visitQuotedNameExpr(self, expr):
        # Quoted names don't accept levels yet.
        return Term(Variable(expr.expression.lexeme[1:-1]))
