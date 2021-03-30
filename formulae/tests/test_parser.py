from formulae.expr import Assign, Grouping, Binary, Unary, Call, Variable, QuotedName, Literal
from formulae.parser import Parser
from formulae.scanner import Scanner
from formulae.token import Token


def parse(x, add_intercept=True):
    return Parser(Scanner(x).scan(add_intercept)).parse()


def test_parse_literal():
    p = parse("'A'")
    assert p == Binary(Literal(1), Token("PLUS", "+"), Literal("A"))

    p = parse("1")
    assert p == Binary(Literal(1), Token("PLUS", "+"), Literal(1))

    p = parse("1.132")
    assert p == Binary(Literal(1), Token("PLUS", "+"), Literal(1.132))


def test_parse_quoted_name():
    p = parse("`$$##!!`")
    assert p == Binary(Literal(1), Token("PLUS", "+"), QuotedName(Token("BQNAME", "`$$##!!`")))


def test_parse_variable():
    p = parse("x")
    assert p == Binary(Literal(1), Token("PLUS", "+"), Variable(Token("IDENTIFIER", "x")))


def test_parse_call():
    p = parse("f(x)")
    assert p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Call(Variable(Token("IDENTIFIER", "f")), [Variable(Token("IDENTIFIER", "x"))]),
    )
    p = parse("module.f(x)")
    assert p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Call(Variable(Token("IDENTIFIER", "module.f")), [Variable(Token("IDENTIFIER", "x"))]),
    )
    p = parse("{x + y}")
    p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Call(
            Variable(Token("IDENTIFIER", "I")),
            [
                Binary(
                    Variable(Token("IDENTIFIER", "x")),
                    Token("PLUS", "+"),
                    Variable(Token("IDENTIFIER", "y")),
                )
            ],
        ),
    )


def test_parse_binary():
    p = parse("x + y")
    assert p == Binary(
        Binary(Literal(1), Token("PLUS", "+"), Variable(Token("IDENTIFIER", "x"))),
        Token("PLUS", "+"),
        Variable(Token("IDENTIFIER", "y")),
    )

    p = parse("x - y")
    assert p == Binary(
        Binary(Literal(1), Token("PLUS", "+"), Variable(Token("IDENTIFIER", "x"))),
        Token("MINUS", "-"),
        Variable(Token("IDENTIFIER", "y")),
    )

    p = parse("x * y")
    assert p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Binary(
            Variable(Token("IDENTIFIER", "x")),
            Token("STAR", "*"),
            Variable(Token("IDENTIFIER", "y")),
        ),
    )

    p = parse("x / y")
    assert p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Binary(
            Variable(Token("IDENTIFIER", "x")),
            Token("SLASH", "/"),
            Variable(Token("IDENTIFIER", "y")),
        ),
    )

    p = parse("x : y")
    assert p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Binary(
            Variable(Token("IDENTIFIER", "x")),
            Token("COLON", ":"),
            Variable(Token("IDENTIFIER", "y")),
        ),
    )

    p = parse("x ** 2")
    assert p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Binary(Variable(Token("IDENTIFIER", "x")), Token("STAR_STAR", "**"), Literal(2)),
    )

    p = parse("x | y")
    assert p == Binary(
        Binary(Literal(1), Token("PLUS", "+"), Variable(Token("IDENTIFIER", "x"))),
        Token("PIPE", "|"),
        Variable(Token("IDENTIFIER", "y")),
    )

    p = parse("x ~ y")
    assert p == Binary(
        Variable(Token("IDENTIFIER", "x")),
        Token("TILDE", "~"),
        Binary(Literal(1), Token("PLUS", "+"), Variable(Token("IDENTIFIER", "y"))),
    )


def test_parse_grouping():
    p = parse("(x + z)")
    assert p == Binary(
        Literal(1),
        Token("PLUS", "+"),
        Grouping(
            Binary(
                Variable(Token("IDENTIFIER", "x")),
                Token("PLUS", "+"),
                Variable(Token("IDENTIFIER", "z")),
            )
        ),
    )


def test_parse_assign():
    # Needs "add_intercept" False, if not it is "1 + z = x"
    p = parse("x = z", False)
    assert p == Assign(Variable(Token("IDENTIFIER", "x")), Variable(Token("IDENTIFIER", "z")))


def test_parse_intercept_disabled():
    p = parse("x + z", False)
    assert p == Binary(
        Variable(Token("IDENTIFIER", "x")), Token("PLUS", "+"), Variable(Token("IDENTIFIER", "z"))
    )

    p = parse("x ~ z", False)
    assert p == Binary(
        Variable(Token("IDENTIFIER", "x")), Token("TILDE", "~"), Variable(Token("IDENTIFIER", "z"))
    )


def test_parse_complex_expressions():
    p = parse("x*y + u:v")
    ast = Binary(
        Binary(
            Literal(1),
            Token("PLUS", "+"),
            Binary(
                Variable(Token("IDENTIFIER", "x")),
                Token("STAR", "*"),
                Variable(Token("IDENTIFIER", "y")),
            ),
        ),
        Token("PLUS", "+"),
        Binary(
            Variable(Token("IDENTIFIER", "u")),
            Token("COLON", ":"),
            Variable(Token("IDENTIFIER", "v")),
        ),
    )
    assert p == ast

    p = parse("y ~ x*z + (x|g)")
    ast = Binary(
        Variable(Token("IDENTIFIER", "y")),
        Token("TILDE", "~"),
        Binary(
            Binary(
                Literal(1),
                Token("PLUS", "+"),
                Binary(
                    Variable(Token("IDENTIFIER", "x")),
                    Token("STAR", "*"),
                    Variable(Token("IDENTIFIER", "z")),
                ),
            ),
            Token("PLUS", "+"),
            Grouping(
                Binary(
                    Variable(Token("IDENTIFIER", "x")),
                    Token("PIPE", "|"),
                    Variable(Token("IDENTIFIER", "g")),
                )
            ),
        ),
    )

    assert p == ast

    p = parse("np.log(y) ~ (`var 1` + `var 2` + `var 3`) ** 3 - `var 2`:`var 3`")
    ast = Binary(
        Call(Variable(Token("IDENTIFIER", "np.log")), [Variable(Token("IDENTIFIER", "y"))]),
        Token("TILDE", "~"),
        Binary(
            Binary(
                Literal(1),
                Token("PLUS", "+"),
                Binary(
                    Grouping(
                        Binary(
                            Binary(
                                QuotedName(Token("BQNAME", "`var 1`")),
                                Token("PLUS", "+"),
                                QuotedName(Token("BQNAME", "`var 2`")),
                            ),
                            Token("PLUS", "+"),
                            QuotedName(Token("BQNAME", "`var 3`")),
                        )
                    ),
                    Token("STAR_STAR", "**"),
                    Literal(3),
                ),
            ),
            Token("MINUS", "-"),
            Binary(
                QuotedName(Token("BQNAME", "`var 2`")),
                Token("COLON", ":"),
                QuotedName(Token("BQNAME", "`var 3`")),
            ),
        ),
    )

    assert p == ast
