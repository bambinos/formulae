class Expr:
    """Abstract class"""
    
    def __init__(self):
        raise ValueError("Abstract class!!!")
    
    def accept(self):
        pass
    
class Grouping(Expr):
    
    def __init__(self, expression):
        self.expression = expression
    
    def accept(self, visitor):
        return visitor.visitGroupingExpr(self)

    
class Binary(Expr):
    
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right
    
    def accept(self, visitor):
        return visitor.visitBinaryExpr(self)

class Unary(Expr):
    
    def __init__(self, operator, right):
        self.operator = operator
        self.right = right
    
    def accept(self, visitor):
        return visitor.visitUnaryExpr(self)

class Call(Expr):
    """Represents built-in or added functions in formulae"""
    
    def __init__(self, callee, arguments):
        self.callee = callee
        self.arguments = arguments
    
    def accept(self, visitor):
        return visitor.visitCallExpr(self)

class Variable(Expr):
    
    def __init__(self, name):
        self.name = name
    
    def accept(self, visitor):
        return visitor.visitVariableExpr(self)
    
class Literal(Expr):
    
    def __init__(self, value):
        self.value = value
    
    def accept(self, visitor):
        return visitor.visitLiteralExpr(self)
