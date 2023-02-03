import numpy as np
import pandas as pd

from formulae import design_matrices
from formulae.transforms import register_stateful_transform


def test_register_stateful_transform():
    @register_stateful_transform
    class Transform:
        __transform_name__ = "transform"

        def __init__(self):
            self.center = None
            self.params_set = False

        def __call__(self, values):
            if not self.params_set:
                self.center = sum(values) / len(values)
                self.params_set = True
            return values - self.center

    df = pd.DataFrame({"x": list(range(10))})
    dm = design_matrices("0 + transform(x)", df)

    assert all(dm.common.__array__().flatten() == np.array(range(10)) - np.mean(range(10)))
