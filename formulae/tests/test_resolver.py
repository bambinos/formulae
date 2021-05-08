import pytest

from formulae.model_description import model_description
from formulae.terms import Variable, Call, Intercept, Term, GroupSpecificTerm, Model, Response
from formulae.terms.call_resolver import LazyCall, LazyVariable

# TODO:
# test repeated terms
# tests for CallTerms, ModelTerms, GroupSpecificTerm and so on...


def test_empty_model_terms():
    # This does not raise an error here, but raises an error in `model_terms()`
    assert model_description("-1") == Model()


def test_term():
    desc = model_description("x")
    comp = Model(Intercept(), Term(Variable("x")))
    assert desc == comp

    desc = model_description("term_name_abc")
    comp = Model(Intercept(), Term(Variable("term_name_abc")))
    assert desc == comp

    desc = model_description("`$%!N4m3##!! NNN`")
    comp = Model(Intercept(), Term(Variable("$%!N4m3##!! NNN")))
    assert desc == comp


def test_term_add():
    desc = model_description("x + y")
    comp = Model(Intercept(), Term(Variable("x")), Term(Variable("y")))
    assert desc == comp

    desc = model_description("x + 5")
    comp = Model(Intercept(), Term(Variable("x")), Term(Variable(5)))
    assert desc == comp

    desc = model_description("x + f(x)")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Call(LazyCall("f", [LazyVariable("x")], {}))),
    )
    assert desc == comp

    desc = model_description("x + y:z")
    comp = Model(Intercept(), Term(Variable("x")), Term(Variable("y"), Variable("z")))
    assert desc == comp

    desc = model_description("x + (1|g)")
    comp = Model(
        Intercept(), Term(Variable("x")), GroupSpecificTerm(Intercept(), Term(Variable("g")))
    )
    assert desc == comp

    desc = model_description("x + (z + y)")
    comp = Model(Intercept(), Term(Variable("x")), Term(Variable("z")), Term(Variable("y")))
    assert desc == comp


def test_term_remove():
    desc = model_description("x - y")
    comp = Model(Intercept(), Term(Variable("x")))
    assert desc == comp

    desc = model_description("x - 5")
    comp = Model(Intercept(), Term(Variable("x")))
    assert desc == comp

    desc = model_description("x - f(x)")
    comp = Model(Intercept(), Term(Variable("x")))
    assert desc == comp

    desc = model_description("x - y:z")
    comp = Model(Intercept(), Term(Variable("x")))
    assert desc == comp

    desc = model_description("x - (1|g)")
    comp = Model(Intercept(), Term(Variable("x")))
    assert desc == comp

    desc = model_description("x - (z + y)")
    comp = Model(Intercept(), Term(Variable("x")))
    assert desc == comp


def test_term_interaction():
    desc = model_description("x:y")
    comp = Model(Intercept(), Term(Variable("x"), Variable("y")))
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x:5")

    desc = model_description("x:f(x)")
    comp = Model(
        Intercept(),
        Term(Variable("x"), Call(LazyCall("f", [LazyVariable("x")], {}))),
    )
    assert desc == comp

    desc = model_description("x:y:z")
    comp = Model(Intercept(), Term(Variable("x"), Variable("y"), Variable("z")))
    assert desc == comp

    desc = model_description("x:y*z")
    comp = Model(
        Intercept(),
        Term(Variable("x"), Variable("y")),
        Term(Variable("z")),
        Term(Variable("x"), Variable("y"), Variable("z")),
    )
    assert desc == comp

    # Note the parenthesis, here `*` resolves earlier than `:`
    desc = model_description("x:(y*z)")
    comp = Model(
        Intercept(),
        Term(Variable("x"), Variable("y")),
        Term(Variable("x"), Variable("z")),
        Term(Variable("x"), Variable("y"), Variable("z")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x:(1|g)")

    desc = model_description("x:(z + y)")
    comp = Model(
        Intercept(),
        Term(Variable("x"), Variable("z")),
        Term(Variable("x"), Variable("y")),
    )
    assert desc == comp


def test_term_power_interaction():
    desc = model_description("x*y")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("y")),
        Term(Variable("x"), Variable("y")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x:5")

    desc = model_description("x*f(x)")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Call(LazyCall("f", [LazyVariable("x")], {}))),
        Term(Variable("x"), Call(LazyCall("f", [LazyVariable("x")], {}))),
    )
    assert desc == comp

    desc = model_description("x*y:z")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("y"), Variable("z")),
        Term(Variable("x"), Variable("y"), Variable("z")),
    )
    assert desc == comp

    desc = model_description("x*y*z")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("y")),
        Term(Variable("x"), Variable("y")),
        Term(Variable("z")),
        Term(Variable("x"), Variable("z")),
        Term(Variable("y"), Variable("z")),
        Term(Variable("x"), Variable("y"), Variable("z")),
    )
    assert desc == comp

    # Note the parenthesis, here `*` resolves earlier than `:`
    desc = model_description("x*(y*z)")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("y")),
        Term(Variable("z")),
        Term(Variable("y"), Variable("z")),
        Term(Variable("x"), Variable("y")),
        Term(Variable("x"), Variable("z")),
        Term(Variable("x"), Variable("y"), Variable("z")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x*(1|g)")

    desc = model_description("x*(z + y)")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("z")),
        Term(Variable("y")),
        Term(Variable("x"), Variable("z")),
        Term(Variable("x"), Variable("y")),
    )
    assert desc == comp


def test_term_slash():
    desc = model_description("x / y")
    comp = Model(Intercept(), Term(Variable("x")), Term(Variable("x"), Variable("y")))
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x / 5")

    desc = model_description("x / f(x)")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("x"), Call(LazyCall("f", [LazyVariable("x")], {}))),
    )
    assert desc == comp

    desc = model_description("x / y:z")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("x"), Variable("y"), Variable("z")),
    )
    assert desc == comp

    with pytest.raises(TypeError):
        model_description("x / (1|g)")

    desc = model_description("x / (z + y)")
    comp = Model(
        Intercept(),
        Term(Variable("x")),
        Term(Variable("x"), Variable("z")),
        Term(Variable("x"), Variable("y")),
    )
    assert desc == comp


def test_group_specific_interactions():

    desc = model_description("0 + (a*b|h+g)")
    comp = Model(
        GroupSpecificTerm(expr=Intercept(), factor=Term(Variable("h"))),
        GroupSpecificTerm(expr=Intercept(), factor=Term(Variable("g"))),
        GroupSpecificTerm(expr=Term(Variable("a")), factor=Term(Variable("h"))),
        GroupSpecificTerm(expr=Term(Variable("a")), factor=Term(Variable("g"))),
        GroupSpecificTerm(expr=Term(Variable("b")), factor=Term(Variable("h"))),
        GroupSpecificTerm(expr=Term(Variable("b")), factor=Term(Variable("g"))),
        GroupSpecificTerm(expr=Term(Variable("a"), Variable("b")), factor=Term(Variable("h"))),
        GroupSpecificTerm(Term(Variable("a"), Variable("b")), factor=Term(Variable("g"))),
    )
    assert desc == comp

    desc = model_description("0 + (0 + a*b|h+g)")
    comp = Model(
        GroupSpecificTerm(expr=Term(Variable("a")), factor=Term(Variable("h"))),
        GroupSpecificTerm(expr=Term(Variable("a")), factor=Term(Variable("g"))),
        GroupSpecificTerm(expr=Term(Variable("b")), factor=Term(Variable("h"))),
        GroupSpecificTerm(expr=Term(Variable("b")), factor=Term(Variable("g"))),
        GroupSpecificTerm(expr=Term(Variable("a"), Variable("b")), factor=Term(Variable("h"))),
        GroupSpecificTerm(Term(Variable("a"), Variable("b")), factor=Term(Variable("g"))),
    )
    assert desc == comp


def test_subset_index():

    desc = model_description("threecats['b'] ~ continuous + dummy")
    comp = Model(
        Intercept(),
        Term(Variable("continuous")),
        Term(Variable("dummy")),
    )
    comp.add_response(Response(Term(Variable("threecats", level="b"))))
    assert desc == comp
