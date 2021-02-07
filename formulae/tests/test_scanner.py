from formulae.scanner import Scanner
from formulae.token import Token

def test_scan_literal():
    sc = Scanner('x').scan()
    comp = [Token('NUMBER', 1, 1), Token('PLUS', '+'), Token('IDENTIFIER', 'x'), Token('EOF', '')]
    assert all([True for i, j in zip(sc, comp) if i == j])