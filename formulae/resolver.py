from .terms import Term, InteractionTerm, ResponseTerm

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
        elif otype == 'COLON':
            return expr.left.accept(self) * expr.right.accept(self)
            
        else:
            pass
    
    def visitUnaryExpr(self):
        self.expr
        pass

    def visitCallExpr(self):
        pass

    def visitVariableExpr(self, expr):
        return Term(expr.name.lexeme, expr.name.lexeme)

    def visitLiteralExpr(self):
        pass
