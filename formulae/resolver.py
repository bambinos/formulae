class Resolver:

    def __init__(self, expr):
        self.expr = expr

    def resolve(self):
        self.expr.accept(self)

    def visitGroupingExpr(self):
        pass

    def visitBinaryExpr(self):
        pass

    def visitUnaryExpr(self):
        self.expr
        pass

    def visitCallExpr(self):
        pass

    def visitVariableExpr(self):
        pass

    def visitLiteralExpr(self):
        pass
