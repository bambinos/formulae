class Token:
    """Representation of a single Token"""

    def __init__(self, _type, lexeme, literal=None):
        self.type = _type
        self.lexeme = lexeme
        self.literal = literal

    def __hash__(self):
        return hash((self.type, self.lexeme, self.literal))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return (
            self.type == other.type
            and self.lexeme == other.lexeme
            and self.literal == other.literal
        )

    def __repr__(self):
        string_list = [
            "'type': " + str(self.type),
            "'lexeme': " + str(self.lexeme),
            "'literal': " + str(self.literal),
        ]
        return "{" + ", ".join(string_list) + "}"

    def __str__(self):
        string_list = [
            "type= " + str(self.type),
            "lexeme= " + str(self.lexeme),
            "literal= " + str(self.literal),
        ]
        return "Token(" + ", ".join(string_list) + ")"
