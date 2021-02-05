import numpy as np

def I(x):
    return x

def center(x):
    return x - np.mean(x)

def scale(x):
    return (x - np.mean(x)) / np.std(x)

TRANSFORMATIONS = {
    "I": I,
    "center": center,
    "scale":scale,
    "standardize": scale
}