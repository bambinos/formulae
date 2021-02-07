import pytest

from formulae.model_description import model_description
from formulae.token import Token

from formulae.expr import *
from formulae.terms import *


# TODO:
# test repeated terms
# tests for CallTerms, ModelTerms, GroupSpecTerm and so on...


def test_empty_model_terms():
    # This does not raise an error here, but raises an error in `model_terms()`
    assert model_description("-1") == ModelTerms()


def test_term():
    desc = model_description("x")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"))
    assert desc == comp

    desc = model_description("term_name_abc")
    comp = ModelTerms(InterceptTerm(), Term("term_name_abc", "term_name_abc"))
    assert desc == comp

    desc = model_description("`$%!N4m3##!! NNN`")
    comp = ModelTerms(InterceptTerm(), Term("$%!N4m3##!! NNN", "$%!N4m3##!! NNN"))
    assert desc == comp


def test_term_add():
    desc = model_description("x + y")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"), Term("y", "y"))
    assert desc == comp

    desc = model_description("x + 5")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"), LiteralTerm(5))
    assert desc == comp

    desc = model_description("x + f(x)")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        CallTerm(
            Call(Variable(Token("IDENTIFIER", "f")), [Variable(Token("IDENTIFIER", "x"))], False)
        ),
    )
    assert desc == comp

    desc = model_description("x + y:z")
    comp = ModelTerms(
        InterceptTerm(), Term("x", "x"), InteractionTerm(Term("y", "y"), Term("z", "z"))
    )
    assert desc == comp

    desc = model_description("x + (1|g)")
    comp = ModelTerms(
        InterceptTerm(), Term("x", "x"), GroupSpecTerm(InterceptTerm(), Term("g", "g"))
    )
    assert desc == comp

    desc = model_description("x + (z + y)")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"), Term("z", "z"), Term("y", "y"))
    assert desc == comp


def test_term_remove():
    desc = model_description("x - y")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"))
    assert desc == comp

    desc = model_description("x - 5")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"))
    assert desc == comp

    desc = model_description("x - f(x)")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"))
    assert desc == comp

    desc = model_description("x - y:z")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"))
    assert desc == comp

    desc = model_description("x - (1|g)")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"))
    assert desc == comp

    desc = model_description("x - (z + y)")
    comp = ModelTerms(InterceptTerm(), Term("x", "x"))
    assert desc == comp


# with pytest.raises(ScanError):


def test_term_interaction():
    desc = model_description("x:y")
    comp = ModelTerms(InterceptTerm(), InteractionTerm(Term("x", "x"), Term("y", "y")))
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x:5")

    desc = model_description("x:f(x)")
    comp = ModelTerms(
        InterceptTerm(),
        InteractionTerm(
            Term("x", "x"),
            CallTerm(
                Call(
                    Variable(Token("IDENTIFIER", "f")), [Variable(Token("IDENTIFIER", "x"))], False
                )
            ),
        ),
    )
    assert desc == comp

    desc = model_description("x:y:z")
    comp = ModelTerms(
        InterceptTerm(), InteractionTerm(Term("x", "x"), Term("y", "y"), Term("z", "z"))
    )
    assert desc == comp

    desc = model_description("x:y*z")
    comp = ModelTerms(
        InterceptTerm(),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
        Term("z", "z"),
        InteractionTerm(Term("x", "x"), Term("y", "y"), Term("z", "z")),
    )
    assert desc == comp

    # Note the parenthesis, here `*` resolves earlier than `:`
    desc = model_description("x:(y*z)")
    comp = ModelTerms(
        InterceptTerm(),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
        InteractionTerm(Term("x", "x"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y"), Term("z", "z")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x:(1|g)")

    desc = model_description("x:(z + y)")
    comp = ModelTerms(
        InterceptTerm(),
        InteractionTerm(Term("x", "x"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
    )
    assert desc == comp


def test_term_power_interaction():
    desc = model_description("x*y")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        Term("y", "y"),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x:5")

    desc = model_description("x*f(x)")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        CallTerm(
            Call(Variable(Token("IDENTIFIER", "f")), [Variable(Token("IDENTIFIER", "x"))], False)
        ),
        InteractionTerm(
            Term("x", "x"),
            CallTerm(
                Call(
                    Variable(Token("IDENTIFIER", "f")), [Variable(Token("IDENTIFIER", "x"))], False
                )
            ),
        ),
    )
    assert desc == comp

    desc = model_description("x*y:z")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        InteractionTerm(Term("y", "y"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y"), Term("z", "z")),
    )
    assert desc == comp

    desc = model_description("x*y*z")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        Term("y", "y"),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
        Term("z", "z"),
        InteractionTerm(Term("x", "x"), Term("z", "z")),
        InteractionTerm(Term("y", "y"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y"), Term("z", "z")),
    )
    assert desc == comp

    # Note the parenthesis, here `*` resolves earlier than `:`
    desc = model_description("x*(y*z)")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        Term("y", "y"),
        Term("z", "z"),
        InteractionTerm(Term("y", "y"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
        InteractionTerm(Term("x", "x"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y"), Term("z", "z")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x*(1|g)")

    desc = model_description("x*(z + y)")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        Term("z", "z"),
        Term("y", "y"),
        InteractionTerm(Term("x", "x"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
    )
    assert desc == comp


def test_term_slash():
    desc = model_description("x / y")
    comp = ModelTerms(
        InterceptTerm(), Term("x", "x"), InteractionTerm(Term("x", "x"), Term("y", "y"))
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x / 5")

    desc = model_description("x / f(x)")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        InteractionTerm(
            Term("x", "x"),
            CallTerm(
                Call(
                    Variable(Token("IDENTIFIER", "f")), [Variable(Token("IDENTIFIER", "x"))], False
                )
            ),
        ),
    )
    assert desc == comp

    desc = model_description("x / y:z")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        InteractionTerm(Term("y", "y"), Term("z", "z"), Term("x", "x")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x / (1|g)")

    desc = model_description("x / (z + y)")
    comp = ModelTerms(
        InterceptTerm(),
        Term("x", "x"),
        InteractionTerm(Term("x", "x"), Term("z", "z")),
        InteractionTerm(Term("x", "x"), Term("y", "y")),
    )
    assert desc == comp
