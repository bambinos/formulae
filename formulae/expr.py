class Expr:
    """Abstract class"""

    def __init__(self):
        raise ValueError("Abstract class")

    def accept(self):
        pass


class Assign(Expr):
    """Expr for Asssignments.

    This type of expressions can be parsed anywhere, but can only be resolved
    within function call arguments.
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __hash__(self):
        return hash((self.name, self.value))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name and self.value == other.value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        right = "  ".join(str(self.value).splitlines(True))
        return f"Assign(\n  name= {self.name}, \n  value= {right}\n)"

    def accept(self, visitor):
        return visitor.visitAssignExpr(self)


class Grouping(Expr):
    def __init__(self, expression):
        self.expression = expression

    def __hash__(self):
        return hash((self.expression, self.expression))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.expression == other.expression

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Grouping(\n  " + "  ".join(str(self.expression).splitlines(True)) + "\n)"

    def accept(self, visitor):
        return visitor.visitGroupingExpr(self)


class Binary(Expr):
    def __init__(self, left, operator, right):
        self.left = left
        self.operator = operator
        self.right = right

    def __hash__(self):
        return hash((self.left, self.operator, self.right))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.left == other.left
            and self.operator == other.operator
            and self.right == other.right
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        left = "  ".join(str(self.left).splitlines(True))
        right = "  ".join(str(self.right).splitlines(True))
        string_list = ["left=" + left, "op=" + str(self.operator.lexeme), "right=" + right]
        return "Binary(\n  " + ",\n  ".join(string_list) + "\n)"

    def accept(self, visitor):
        return visitor.visitBinaryExpr(self)


class Unary(Expr):
    def __init__(self, operator, right):
        self.operator = operator
        self.right = right

    def __hash__(self):
        return hash((self.operator, self.right))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.operator == other.operator and self.right == other.right

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        right = "  ".join(str(self.right).splitlines(True))
        string_list = ["op=" + str(self.operator.lexeme), "right=" + right]
        return "Unary(\n  " + ", ".join(string_list) + "\n)"

    def accept(self, visitor):
        return visitor.visitUnaryExpr(self)


class Call(Expr):
    """Function call expressions"""

    def __init__(self, callee, args, special=False):
        self.callee = callee
        self.args = args
        self.special = special

    def __hash__(self):
        return hash((self.callee, self.args, self.special))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.callee == other.callee
            and self.args == other.args
            and self.special == other.special
        )

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = [
            "callee=" + str(self.callee),
            "args=" + "  ".join(str(self.args).splitlines(True)),
            "special=" + str(self.special),
        ]
        return "Call(\n  " + ",\n  ".join(string_list) + "\n)"

    def accept(self, visitor):
        return visitor.visitCallExpr(self)


class Variable(Expr):
    def __init__(self, name, level=None):
        self.name = name
        self.level = level

    def __hash__(self):
        return hash((self.name, self.level))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name and self.level == other.level

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_list = ["name=" + self.name.lexeme]
        if self.level is not None:
            string_list.append("level=" + self.level.lexeme)
        return "Variable(" + ",\n  ".join(string_list) + ")"

    def accept(self, visitor):
        return visitor.visitVariableExpr(self)


class QuotedName(Expr):
    """Expressions for back-quoted names (i.e. `@1wrid_name!!`)"""

    def __init__(self, expression):
        self.expression = expression

    def __hash__(self):
        return hash((self.expression))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.expression == other.expression

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "QuotedName(" + self.expression.lexeme + ")"

    def accept(self, visitor):
        return visitor.visitQuotedNameExpr(self)


class Literal(Expr):
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash((self.value))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.value == other.value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Literal(" + str(self.value) + ")"

    def accept(self, visitor):
        return visitor.visitLiteralExpr(self)
