from .expr import (Grouping, Binary, Unary, Call, Variable, Literal)
from .utils import listify

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
        return self.peek().type in listify(types)

    def match(self, types):
        if self.check(types):
            self.advance()
            return True
        else:
            return False

    def consume(self, _type, message):
        """Consumes the next Token

        First, it checks if the next Token is of the expected type.
        If True, it calls self.advance() and it's Saul Goodman.
        Otherwise, we've found an error.
        """
        if self.check(_type):
            return self.advance()
        else:
            raise ParseError(message)

    # Here we start walking through the grammar
    def parse(self):
        return self.expression()

    def expression(self, is_arg=False):
        return self.tilde(is_arg)
        
    def tilde(self, is_arg=False):
        expr = self.addition(is_arg)
        if self.match('TILDE'):
            operator = self.previous()
            right = self.addition(is_arg)
            expr = Binary(expr, operator, right, is_arg)
        return expr

    def addition(self, is_arg):
        expr = self.multiplication(is_arg)
        while self.match(['MINUS', 'PLUS']):
            operator = self.previous()
            right = self.multiplication(is_arg)
            expr = Binary(expr, operator, right, is_arg)
        return expr
    
    def multiplication(self, is_arg=False):
        expr = self.unary(is_arg)
        while self.match(['STAR', 'STAR_STAR', 'SLASH', 'COLON']):
            operator = self.previous()
            right = self.unary(is_arg)
            expr = Binary(expr, operator, right, is_arg)
        return expr

    def unary(self, is_arg=False):
        if self.match(['PLUS', 'MINUS']):
            operator = self.previous()
            right = self.unary(is_arg)
            return Unary(operator, right, is_arg)
        return self.call(is_arg)

    def call(self, is_arg=False):
        expr = self.primary()
        while True:
            if self.match('LEFT_PAREN'):
                expr = self.finishcall(expr)
            else:
                break
        return expr

    def finishcall(self, callee, is_arg=False):
        # TODO: Check custom calls
        arguments = []
        if not self.check('RIGHT_PAREN'):
            while True:
                # TODO: check args len?
                arguments.append(self.expression(is_arg=True))
                if not self.match('COMMA'):
                    break
        self.consume('RIGHT_PAREN', "Expect ')' after arguments.")
        # The is_arg enables nested calls
        return Call(callee, arguments, is_arg)

    def primary(self, is_arg=False):
        if self.match('NUMBER'):
            return Literal(self.previous().literal, is_arg)
        elif self.match('IDENTIFIER'):
            return Variable(self.previous(), is_arg)
        elif self.match('LEFT_PAREN'):
            expr = self.expression(is_arg)
            if self.match('PIPE'):
                operator = self.previous()
                right = self.call()
                self.consume('RIGHT_PAREN', "Expect ')' after expression.")
                # probably this pipe will fail if is_arg==True
                return Binary(expr, operator, right, is_arg) 
            self.consume('RIGHT_PAREN', "Expect ')' after expression.")
            return Grouping(expr, is_arg)
        else:
            raise ParseError("Expect expression.")


# ':' and '*' must be left-associative
# 
