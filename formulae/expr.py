class Expr:
    """Abstract class"""

    def __init__(self):
        raise ValueError("Abstract class!!!")

    def accept(self):
        pass

class Grouping(Expr):

    def __init__(self, expression):
        self.expression = expression

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Grouping(' + str(self.expression) + ')'

    def accept(self, visitor):
        return visitor.visitGroupingExpr(self)


class Binary(Expr):

    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "left=" + str(self.left),
            "op=" + str(self.operator.lexeme),
            "right=" + str(self.right)
        ]
        return 'Binary(' + ', '.join(string_list) + ')'

    def accept(self, visitor):
        return visitor.visitBinaryExpr(self)

class Unary(Expr):

    def __init__(self, operator, right):
        self.operator = operator
        self.right = right

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "op=" + str(self.operator.lexeme),
            "right=" + str(self.right)
        ]
        return 'Unary(' + ', '.join(string_list) + ')'

    def accept(self, visitor):
        return visitor.visitUnaryExpr(self)

class Call(Expr):
    """Represents built-in or added functions in formulae"""

    def __init__(self, callee, arguments):
        self.callee = callee
        self.arguments = arguments

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "callee=" + str(self.callee),
            "args=" + ", ".join([repr(arg) for arg in self.arguments])
        ]
        return 'Call(' + ', '.join(string_list) + ')'

    def accept(self, visitor):
        return visitor.visitCallExpr(self)

class Variable(Expr):

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Variable('+ self.name.lexeme + ')'

    def accept(self, visitor):
        return visitor.visitVariableExpr(self)

class Literal(Expr):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Literal('+ str(self.value) + ')'

    def accept(self, visitor):
        return visitor.visitLiteralExpr(self)
