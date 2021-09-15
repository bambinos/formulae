<img src="docs/logo/formulae_large.png" width=250></img>

[![PyPI version](https://badge.fury.io/py/formulae.svg)](https://badge.fury.io/py/formulae)
[![codecov](https://codecov.io/gh/bambinos/formulae/branch/master/graph/badge.svg)](https://codecov.io/gh/bambinos/formulae)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

# formulae

formulae is a Python library that implements Wilkinson's formulas for mixed-effects models. The main difference with other implementations like [Patsy](https://github.com/pydata/patsy) or [formulaic](https://github.com/matthewwardrop/formulaic) is that formulae can work with formulas describing a model with both common and group specific effects (a.k.a. fixed and random effects, respectively).

This package has been written to make it easier to specify models with group effects in [Bambi](https://github.com/bambinos/bambi), a package that makes it easy to work with Bayesian GLMMs in Python, but it could be used independently as a backend for another library. The approach in this library is to extend classical statistical formulas in a similar way than in R package [lme4](https://CRAN.R-project.org/package=lme4).

## Installation

formulae requires a working Python interpreter (3.7+) and the libraries numpy, scipy and pandas with versions specified in the [requirements.txt](https://github.com/bambinos/formulae/blob/master/requirements.txt) file.

Assuming a standard Python environment is installed on your machine (including pip), the latest release of formulae can be installed in one line using pip:

`pip install formulae`

Alternatively, if you want the development version of the package you can install from GitHub:

`pip install git+https://github.com/bambinos/formulae.git`

## Documentation

The official documentation can be found [here](https://bambinos.github.io/formulae)

## Notes

- The `data` argument only accepts objects of class `pandas.DataFrame`.
- `y ~ .` is not implemented and won't be implemented in a first version. However, it is planned to be included in the future.
