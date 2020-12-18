class ParseError(Exception):
    pass

class Parser:
    """Consume sequences of Tokens"""

    def __init__(self, tokens):
        self.current = 0
        self.tokens = tokens
        # pass options to understand custom functionality

    def at_end(self):
        return self.peek().type == 'EOF'

    def advance(self):
        if not self.at_end():
            self.current += 1
            return self.tokens[self.current - 1]

    def peek(self):
        """Returns the Token we are about to consume"""
        return self.tokens[self.current]

    def previous(self):
        """Returns the last Token we consumed"""
        return self.tokens[self.current - 1]

    def check(self, types):
        # Checks multiple types at once
        if self.at_end():
            return False
        return self.peek().type in types

    def match(types):
        if self.check(types):
            self.advance()
            return True
        else:
            return False

    def consume(_type, message):
        """Consumes the next Token

        First, it checks if the next Token is of the expected type.
        If True, it calls self.advance() and it's Saul Goodman.
        Otherwise, we've found an error.
        """
        if self.check(_type, message):
            return self.advance()
        else:
            raise ParseError(message)

    def parse(self):
        self.expression()
