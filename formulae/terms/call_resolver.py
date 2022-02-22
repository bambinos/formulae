import operator

from formulae.expr import Assign
from formulae.transforms import STATEFUL_TRANSFORMS


class CallResolverError(Exception):
    pass


class LazyOperator:
    """Unary and Binary lazy operators.

    Functions calls like ``a + b`` are converted into a LazyOperator that is resolved when you
    explicitly evaluates it.

    Parameters
    ----------
    op: builtin_function_or_method
        An operator in the ``operator`` built-in module. It can be one of ``add``, ``pos``, ``sub``,
        ``neg``, ``pow``, ``mul``, and ``truediv``.
    args:
        One or two lazy instances.
    """

    SYMBOLS = {
        "add": "+",
        "pos": "+",
        "sub": "-",
        "neg": "-",
        "pow": "**",
        "mul": "*",
        "truediv": "/",
        "eq": "==",
        "ne": "!=",
        "le": "<=",
        "lt": "<",
        "ge": ">=",
        "gt": ">",
    }

    def __init__(self, op, *args):
        self.op = op
        self.args = args
        self.symbol = self.SYMBOLS[op.__name__]

    def __str__(self):
        if len(self.args) == 1:
            return f"{self.symbol}{self.args[0]}"
        else:
            return f"{self.args[0]} {self.symbol} {self.args[1]}"

    def __hash__(self):
        return hash((self.symbol, *self.args))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.symbol == other.symbol and set(self.args) == set(other.args)

    def accept(self, visitor):
        return visitor.visitLazyOperator(self)

    def eval(self, data_mask, env):
        """Evaluates the operation.

        Evaluates the arguments involved in the operation, calls the Python operator, and returns
        the result.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.

        Returns
        -------
        result:
            The value obtained from the operator call.
        """
        return self.op(*[arg.eval(data_mask, env) for arg in self.args])


class LazyVariable:
    """Lazy variable name.

    The variable represented in this object does not hold any value until it is explicitly evaluated
    within a data mask and an evaluation environment.

    Parameters
    ----------
    name: str
        The name of the variable it represents.
    """

    BUILTINS = {"True": True, "False": False, "None": None}

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.name == other.name

    def accept(self, visitor):
        return visitor.visitLazyVariable(self)

    def eval(self, data_mask, env):
        """Evaluates variable.

        First it looks for the variable in ``data_mask``. If not found there, it looks in
        ``env``. Then it just returns the value the variable represents in either the
        data mask or the evaluation environment.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.

        Returns
        -------
        result:
            The value represented by this name in either the data mask or the environment.
        """
        if self.name in self.BUILTINS:
            result = self.BUILTINS[self.name]
        else:
            try:
                result = data_mask[self.name]
            except KeyError:
                try:
                    result = env.namespace[self.name]
                except KeyError as e:
                    raise e
        return result


class LazyValue:
    """Lazy representation of a value in Python.

    This object holds a value (a string or a number).
    It returns its value only when it is evaluated via ``.eval()``.

    Parameters
    ----------
    value: string or numeric
        The value it holds.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.value == other.value

    def accept(self, visitor):
        return visitor.visitLazyValue(self)

    def eval(self, *args, **kwargs):  # pylint: disable = unused-argument
        """Evaluates the value.

        Simply returns the value. Other arguments are ignored but required for consistency
        among all the lazy objects.

        Returns
        -------
        value:
            The value this obejct represents.
        """
        return self.value


class LazyCall:
    """Lazy representation of a function call.

    This class represents a function that can be a stateful transform (a function with memory)
    whose arguments can also be stateful transforms.

    To evaluate these functions we don't create a string representing Python code and let ``eval()``
    run it. We take care of all the steps of the evaluation to make sure all the possibly nested
    stateful transformations are handled correctly.

    Parameters
    ----------
    callee: string
        The name of the function
    args: list
        A list of lazy objects that are evaluated when calling the function this object represents.
    kwargs: dict
        A dictionary of named arguments that are evaluated when calling the function this object
        represents.
    """

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

    def eval(self, data_mask, env):
        """Evaluate the call.

        This method first evaluates all its arguments, which are themselves lazy objects, and then
        proceeds to evaluate the call it represents.

        Parameters
        ----------
        data_mask: pd.DataFrame
            The data frame where variables are taken from
        env: Environment
            The environment where values and functions are taken from.

        Returns
        -------
        result:
            The result of the call evaluation.
        """
        if self.stateful_transform:
            callee = self.stateful_transform
        else:
            callee = get_function_from_module(self.callee, env)

        args = [arg.eval(data_mask, env) for arg in self.args]
        kwargs = {name: arg.eval(data_mask, env) for name, arg in self.kwargs.items()}

        return callee(*args, **kwargs)


class CallResolver:
    """Visitor that walks an AST representing a regular call and returns a lazy version of it."""

    BINARY_OPERATORS = {
        "PLUS": operator.add,
        "MINUS": operator.sub,
        "STAR_STAR": operator.pow,
        "STAR": operator.mul,
        "SLASH": operator.truediv,
        "EQUAL_EQUAL": operator.eq,
        "BANG_EQUAL": operator.ne,
        "LESS_EQUAL": operator.le,
        "LESS": operator.lt,
        "GREATER_EQUAL": operator.ge,
        "GREATER": operator.gt,
    }

    UNARY_OPERATORS = {"PLUS": operator.pos, "MINUS": operator.neg}

    def __init__(self, expr):
        self.expr = expr

    def resolve(self):
        return self.expr.accept(self)

    def visitGroupingExpr(self, expr):
        return expr.expression.accept(self)

    def visitBinaryExpr(self, expr):
        otype = expr.operator.kind
        op = self.BINARY_OPERATORS.get(otype)
        if op is None:
            raise CallResolverError(f"Can't resolve call with binary expression of type '{otype}'")
        return LazyOperator(op, expr.left.accept(self), expr.right.accept(self))

    def visitUnaryExpr(self, expr):
        otype = expr.operator.kind
        op = self.UNARY_OPERATORS.get(otype)
        if op is None:
            raise CallResolverError(f"Can't resolve call with unary expression of type '{otype}'")
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


def get_function_from_module(name, env):
    names = name.split(".")
    if len(names) == 1:
        fun = env.namespace[names[0]]
    else:
        module_name = names[0]
        function_name = names[-1]
        inner_modules_names = names[1:-1]

        module = env.namespace[module_name]

        if inner_modules_names:
            inner_module = getattr(module, inner_modules_names[0])
            for inner_module_name in inner_modules_names[1:]:
                inner_module = getattr(inner_module, inner_module_name)
            fun = getattr(inner_module, function_name)
        else:
            fun = getattr(module, function_name)
    return fun
