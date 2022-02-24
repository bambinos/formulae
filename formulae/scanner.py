# pylint: disable=relative-beyond-top-level
from .token import Token


class ScanError(Exception):
    pass


class Scanner:
    """Scan formula string and returns Tokens"""

    def __init__(self, code):
        """Scans a model formula and returns a list of Tokens

        Parameters
        ----------
        code : string
            The code to be scanned.
        """
        self.code = code
        self.start = 0
        self.current = 0
        self.tokens = []

        if not len(self.code):
            raise ScanError("'code' is a string of length 0.")

    def at_end(self):
        return self.current >= len(self.code)

    def advance(self):
        self.current += 1
        return self.code[self.current - 1]

    def peek(self):
        if self.at_end():
            return ""
        return self.code[self.current]

    def peek_next(self):
        if self.current + 1 >= len(self.code):
            return ""
        return self.code[self.current + 1]

    def match(self, expected):
        if self.at_end():
            return False
        if self.code[self.current] != expected:
            return False
        self.current += 1
        return True

    def add_token(self, kind, literal=None):
        # Only literals have "literal != None"
        source = self.code[self.start : self.current]
        self.tokens.append(Token(kind, source, literal))

    def scan_token(self):
        char = self.advance()
        if char in ["'", '"']:
            self.char()
        elif char == "(":
            self.add_token("LEFT_PAREN")
        elif char == ")":
            self.add_token("RIGHT_PAREN")
        elif char == "[":
            self.add_token("LEFT_BRACKET")
        elif char == "]":
            self.add_token("RIGHT_BRACKET")
        elif char == "{":
            self.add_token("LEFT_BRACE")
        elif char == "}":
            self.add_token("RIGHT_BRACE")
        elif char == "`":
            self.backquote()
        elif char == ",":
            self.add_token("COMMA")
        elif char == ".":
            if self.peek().isdigit():
                self.floatnum()
            else:
                self.add_token("PERIOD")
        elif char == "+":
            self.add_token("PLUS")
        elif char == "-":
            self.add_token("MINUS")
        elif char == "/":
            if self.match("/"):
                self.add_token("SLASH_SLASH")
            else:
                self.add_token("SLASH")
        elif char == "*":
            if self.match("*"):
                self.add_token("STAR_STAR")
            else:
                self.add_token("STAR")
        elif char == "!":
            if self.match("="):
                self.add_token("BANG_EQUAL")
            else:
                self.add_token("BANG")
        elif char == "=":
            if self.match("="):
                self.add_token("EQUAL_EQUAL")
            else:
                self.add_token("EQUAL")
        elif char == "<":
            if self.match("="):
                self.add_token("LESS_EQUAL")
            else:
                self.add_token("LESS")
        elif char == ">":
            if self.match("="):
                self.add_token("GREATER_EQUAL")
            else:
                self.add_token("GREATER")
        elif char == "%":
            self.add_token("MODULO")
        elif char == "~":
            self.add_token("TILDE")
        elif char == ":":
            self.add_token("COLON")
        elif char == "|":
            self.add_token("PIPE")
        elif char in [" ", "\n", "\t", "\r"]:
            pass
        elif char.isdigit():
            self.number()
        elif char.isalpha():
            self.identifier()
        else:
            raise ScanError("Unexpected character: " + str(char))

    def scan(self, add_intercept=True):
        """Scan formula string.

        Parameters
        ----------
        add_intercept : bool
            Indicates if an implicit intercept should be included. Defaults to True.

        Returns
        -------
        tokens : list
            A list of objects of class Token
        """
        while not self.at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token("EOF", ""))

        # Check number of '~' and add implicit intercept
        tilde_idx = [i for i in range(len(self.tokens)) if is_tilde(self.tokens[i])]

        if len(tilde_idx) > 1:
            raise ScanError("There is more than one '~' in model formula")

        if add_intercept:
            if len(tilde_idx) == 0:
                self.tokens = [Token("NUMBER", "1", 1), Token("PLUS", "+")] + self.tokens
            if len(tilde_idx) == 1:
                self.tokens.insert(tilde_idx[0] + 1, Token("NUMBER", "1", 1))
                self.tokens.insert(tilde_idx[0] + 2, Token("PLUS", "+"))

        return self.tokens

    def floatnum(self):
        while self.peek().isdigit():
            self.advance()
        self.add_token("NUMBER", float(self.code[self.start : self.current]))

    def number(self):
        is_float = False
        while self.peek().isdigit():
            self.advance()
        # Look for fractional part, if present
        if self.peek() == "." and self.peek_next().isdigit():
            is_float = True
            # Consume the dot
            self.advance()
            # Keep consuming numbers, if present
            while self.peek().isdigit():
                self.advance()
        if is_float:
            token = float(self.code[self.start : self.current])
        else:
            token = int(self.code[self.start : self.current])

        self.add_token("NUMBER", token)

    def identifier(self):
        # 'mod.function' is also an identifier
        while self.peek().isalnum() or self.peek() in [".", "_"]:
            self.advance()
        self.add_token("IDENTIFIER")

    def char(self):
        while self.peek() not in ["'", '"'] and not self.at_end():
            self.advance()

        if self.at_end():
            raise ScanError("Unterminated string.")

        # The closing quotation mark.
        self.advance()

        # Trim the surrounding quotes.
        value = self.code[self.start + 1 : self.current - 1]
        self.add_token("STRING", value)

    def backquote(self):
        while True:
            if self.peek() == "`":
                break
            self.advance()
        self.advance()
        self.add_token("BQNAME")


def is_tilde(token):
    return token.kind == "TILDE"
