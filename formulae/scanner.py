from .token import Token

class ScanError(Exception):
    pass

class Scanner:
    """Scan formula string and returns Tokens"""

    def __init__(self, code):
        self.code = code
        self.start = 0
        self.current = 0
        self.tokens = []
        self.keywords = {}

        if not len(self.code):
            raise ScanError("'code' is a string of length 0.")

    def at_end(self):
        return self.current >= len(self.code)

    def advance(self):
        self.current += 1
        return self.code[self.current - 1]

    def peek(self):
        if self.at_end():
            return ''
        return self.code[self.current]

    def peek_next(self):
        if self.current + 1 >= len(self.code): # o len(self.code) + 1
            return ''
        return self.code[self.current + 1]

    def match(self, expected):
        if self.at_end():
            return False
        if self.code[self.current] != expected:
            return False
        self.current += 1
        return True

    def scan_token(self):
        char = self.advance()
        if char == '(':
            self.add_token('LEFT_PAREN')
        elif char == ')':
            self.add_token('RIGHT_PAREN')
        elif char == '[':
            self.add_token('LEFT_BRACKET')
        elif char == ']':
            self.add_token('RIGHT_BRACKET')
        elif char == '{':
            self.add_token('LEFT_BRACE')
        elif char == '}':
            self.add_token('RIGHT_BRACE')
        elif char == ',':
            self.add_token('COMMA')
        elif char == '.':
            self.add_token('PERIOD')
        elif char == '+':
            self.add_token('PLUS')
        elif char == '-':
            self.add_token('MINUS')
        elif char == '/':
            self.add_token('SLASH')
        elif char == '*':
            if self.match('*'):
                self.add_token('STAR_STAR')
            else:
                self.add_token('STAR')
        elif char == '~':
            self.add_token('TILDE')
        elif char == ':':
            self.add_token('COLON')
        elif char == '|':
            self.add_token('PIPE')
        elif char in [' ', '\n', '\t', '\r']:
            return None
        elif char.isdigit():
            self.number()
        elif char.isalpha():
            self.identifier()
        else:
            raise ValueError("Unexpected character: " + str(char))

    def scan_tokens(self):
        while not self.at_end():
            self.start = self.current
            self.scan_token()
        self.tokens.append(Token('EOF', ''))
        return self.tokens

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
            token = float(self.code[self.start:self.current])
        else:
            token = int(self.code[self.start:self.current])

        self.add_token('NUMBER', token)

    def identifier(self):
        while self.peek().isalnum():
            self.advance()
        # Check if the identifier is a reserved word
        identifier = self.code[self.start:self.current]
        if identifier in self.keywords.keys():
            _type = self.keywords[identifier]
        else:
            _type = 'IDENTIFIER'
        self.add_token(_type)

    def add_token(self, _type, literal=None):
        # Only literals have "literal != None"
        source = self.code[self.start:self.current]
        self.tokens.append(Token(_type, source, literal))
