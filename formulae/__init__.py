import logging

from .config import config
from .matrices import design_matrices
from .model_description import model_description
from .version import __version__

__all__ = [
    "config",
    "design_matrices",
    "model_description",
    "__version__",
]

_log = logging.getLogger("formulae")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
