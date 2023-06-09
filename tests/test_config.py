import pytest

from formulae.config import Config


def test_config():
    config = Config()

    assert config["EVAL_NEW_CATEGORIES"] == "error"
    assert config.EVAL_NEW_CATEGORIES == "error"

    with pytest.raises(ValueError, match="anything is not a valid value for 'EVAL_NEW_CATEGORIES'"):
        config.EVAL_NEW_CATEGORIES = "anything"

    with pytest.raises(KeyError, match="'DOESNT_EXIST' is not a valid configuration option"):
        config.DOESNT_EXIST = "anything"


test_config()
