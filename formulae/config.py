class Config:
    FIELDS = {"EVAL_UNSEEN_CATEGORIES": ("error", "warning", "silent")}

    def __init__(self, config_dict: dict = None):
        config_dict = {} if config_dict is None else config_dict
        for field, choices in Config.FIELDS.items():
            if field in config_dict:
                value = config_dict[field]
            else:
                value = choices[0]
            self[field] = value

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __setattr__(self, key, value):
        if key in Config.FIELDS:
            if value in Config.FIELDS[key]:
                super().__setattr__(key, value)
            else:
                raise ValueError(f"{value} is not a valid value for '{key}'")
        else:
            raise KeyError(f"'{key}' is not a valid configuration option")

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):  # pragma: no cover
        lines = []
        for field, choices in Config.FIELDS.items():
            lines.append(f"{field}: {self[field]} (available: {list(choices)})")
        header = ["Formulae configuration"]
        header.append("-" * len(header[0]))
        return "\n".join(header + lines)

    def __repr__(self):  # pragma: no cover
        return str(self)


config = Config()
