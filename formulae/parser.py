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
    
    def expression(self):
        return self.modelization()
    
    def modelization(self):
        expr = self.addition()
        if self.match('TILDE'):
            operator = self.previous()
            right = self.addition()
            expr = Binary(expr, operator, right)
        return expr
    
    def addition(self):
        expr = self.multiplication()
        while self.match(['MINUS', 'PLUS']):
            operator = self.previous()
            right = self.multiplication()
            expr = Binary(expr, operator, right)
        return expr
    
    def multiplication(self):
        # TODO: Remember "/" does not mean division unless within "I()""
        # TODO: COLON and PIPE could use descendant class of Binary operation.
        expr = self.unary()
        while self.match(['STAR', 'STAR_STAR', 'SLASH', 'COLON', 'PIPE']):
            operator = self.previous()
            right = self.unary()
            expr = Binary(expr, operator, right)
        return expr
    
    def unary(self):
        if self.match(['PLUS', 'MINUS']):
            operator = self.previous()
            right = self.unary()
            return Unary(operator, right)
        return self.call()
    
    def call(self):
        expr = self.primary()
        while True:
            if self.match('LEFT_PAREN'):
                expr = self.finishcall(expr)
            else:
                break
        return expr
    
    def finishcall(self, callee):
        # TODO: Check custom calls
        arguments = []
        if not self.check('RIGHT_PAREN'):
            while True:
                # TODO: check args len?
                arguments.append(self.expression())
                if not self.match('COMMA'):
                    break
        self.consume('RIGHT_PAREN', "Expect ')' after arguments.")
        return Call(callee, arguments)
    
    def primary(self):
        if self.match('NUMBER'):
            return Literal(self.previous().literal)
        elif self.match('IDENTIFIER'):
            return Variable(self.previous())
        elif self.match('LEFT_PAREN'):
            expr = self.expression()
            self.consume('RIGHT_PAREN', "Expect ')' after expression.")
            return Grouping(expr)
        else:
            raise ParseError("Expect expression.")
