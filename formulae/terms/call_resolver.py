import operator

from formulae.expr import Assign
from formulae.transforms import STATEFUL_TRANSFORMS


class CallResolverError(Exception):
    pass


class LazyOperator:
    """Unary and Binary lazy operators.

    Functions calls like `a + b` are converted into a LazyOperator that is resolved when you
    explicitly call it like `lazy_operator()`.
    """

    def __init__(self, op, *args):
        self.op = op
        self.args = args
        self.symbol = self._get_symbol()

    def __str__(self):
        if len(self.args) == 1:
            return f"{self.symbol}{self.args[0]}"
        else:
            return f"{self.args[0]} {self.symbol} {self.args[1]}"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.symbol, *self.args))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.symbol == other.symbol and set(self.args) == set(other.args)

    def _get_symbol(self):
        oname = self.op.__name__
        if oname in ["add", "pos"]:
            symbol = "+"
        elif oname in ["sub", "neg"]:
            symbol = "-"
        elif oname == "pow":
            symbol = "**"
        elif oname == "mul":
            symbol = "*"
        elif oname == "truediv":
            symbol = "/"
        return symbol

    def accept(self, visitor):
        return visitor.visitLazyOperator(self)

    def eval(self, data_mask, eval_env):
        return self.op(*[arg.eval(data_mask, eval_env) for arg in self.args])


class LazyVariable:
    """Stores a name that is looked later into objects"""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    def accept(self, visitor):
        return visitor.visitLazyVariable(self)

    def eval(self, data_mask, eval_env):
        try:
            result = data_mask[self.name]
        except KeyError:
            try:
                result = eval_env.namespace[self.name]
            except KeyError as e:
                raise e
        return result


class LazyValue:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash((self.value))

    def accept(self, visitor):
        return visitor.visitLazyValue(self)

    def eval(self, data_mask, eval_env):  # pylint: disable = unused-argument
        return self.value


class LazyCall:
    def __init__(self, callee, args, kwargs):
        self.callee = callee
        self.args = args
        self.kwargs = kwargs
        self.stateful_transform = None

        if self.callee in STATEFUL_TRANSFORMS:
            self.stateful_transform = STATEFUL_TRANSFORMS[self.callee]()

    def __str__(self):
        args = [str(arg) for arg in self.args]
        kwargs = [f"{name} = {str(arg)}" for name, arg in self.kwargs.items()]
        return f"{self.callee}({', '.join(args + kwargs)})"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.callee, *self.args, *self.kwargs))

    def __eq__(self, other):
        return (
            self.callee == other.callee
            and set(self.args) == set(other.args)
            and set(self.kwargs) == set(other.kwargs)
        )

    def accept(self, visitor):
        return visitor.visitLazyCall(self)

    def eval(self, data_mask, eval_env):
        if self.stateful_transform:
            callee = self.stateful_transform
        else:
            callee = eval_env.eval(self.callee)

        args = [arg.eval(data_mask, eval_env) for arg in self.args]
        kwargs = {name: arg.eval(data_mask, eval_env) for name, arg in self.kwargs.items()}

        return callee(*args, **kwargs)


class CallResolver:
    def __init__(self, expr):
        self.expr = expr

    def resolve(self):
        return self.expr.accept(self)

    def visitGroupingExpr(self, expr):
        return expr.expression.accept(self)

    def visitBinaryExpr(self, expr):
        otype = expr.operator.type
        if otype == "PLUS":
            op = operator.add
        elif otype == "MINUS":
            op = operator.sub
        elif otype == "STAR_STAR":
            op = operator.pow
        elif otype == "STAR":
            op = operator.mul
        elif otype == "SLASH":
            op = operator.truediv
        else:
            raise CallResolverError(f"Can't resolve call wih binary expression of type '{otype}'")
        return LazyOperator(op, expr.left.accept(self), expr.right.accept(self))

    def visitUnaryExpr(self, expr):
        otype = expr.operator.type
        if otype == "PLUS":
            op = operator.pos
        elif otype == "MINUS":
            op = operator.neg
        else:
            raise CallResolverError(f"Can't resolve unary expression of type '{otype}'")

        return LazyOperator(op, expr.right.accept(self))

    def visitCallExpr(self, expr):
        args = []
        kwargs = {}
        for arg in expr.args:
            if isinstance(arg, Assign):
                kwargs[arg.name.name.lexeme] = arg.value.accept(self)
            else:
                args.append(arg.accept(self))
        return LazyCall(expr.callee.name.lexeme, args, kwargs)

    def visitVariableExpr(self, expr):
        return LazyVariable(expr.name.lexeme)

    def visitLiteralExpr(self, expr):
        return LazyValue(expr.value)

    def visitQuotedNameExpr(self, expr):
        return LazyVariable(expr.expression.lexeme[1:-1])
