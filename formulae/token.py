class Token:
    def __init__(self, _type, lexeme, literal=None):
        self.type = _type
        self.lexeme = lexeme
        self.literal = literal

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
