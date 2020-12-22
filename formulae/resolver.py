from .terms import Term, InteractionTerm, LiteralTerm, ResponseTerm

class ResolveError(Exception):
    pass

class Resolver:

    def __init__(self, expr):
        self.expr = expr

    def resolve(self):
        return self.expr.accept(self)

    def visitGroupingExpr(self):
        pass

    def visitBinaryExpr(self, expr):
        otype = expr.operator.type
        if otype == 'TILDE':
            return ResponseTerm(expr.left.accept(self)) | expr.right.accept(self)
        if otype == 'PLUS':
            return expr.left.accept(self) | expr.right.accept(self)
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
        else:
            raise ResolveError("Couldn't resolve BinaryExpr with otype '" + otype + "'")
    
    def visitUnaryExpr(self):
        self.expr
        pass

    def visitCallExpr(self):
        pass

    def visitVariableExpr(self, expr):
        return Term(expr.name.lexeme, expr.name.lexeme)

    def visitLiteralExpr(self, expr):
        return LiteralTerm(expr.value)
