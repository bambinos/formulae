class Token:
    """Representation of a single Token"""

    def __init__(self, kind, lexeme, literal=None):
        self.kind = kind
        self.lexeme = lexeme
        self.literal = literal

    def __eq__(self, other):
        return (
            self.kind == other.kind
            and self.lexeme == other.lexeme
            and self.literal == other.literal
        )

    def __repr__(self):  # pragma: no cover
        string_list = [
            "'kind': " + str(self.kind),
            "'lexeme': " + str(self.lexeme),
            "'literal': " + str(self.literal),
        ]
        return "{" + ", ".join(string_list) + "}"

    def __str__(self):  # pragma: no cover
        string_list = [
            "kind= " + str(self.kind),
            "lexeme= " + str(self.lexeme),
            "literal= " + str(self.literal),
        ]
        return "Token(" + ", ".join(string_list) + ")"
