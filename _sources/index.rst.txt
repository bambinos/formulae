Formulae
===================================================
|PyPI version|
|Black|


.. |PyPI version| image:: https://badge.fury.io/py/formulae.svg
    :target: https://badge.fury.io/py/formulae

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black



Formulae is a Python library that implements Wilkinson's formulas for statistical models à la lme4.

This package has been written to make it easier to specify models with group effects in `Bambi <https://github.com/bambinos/bambi>`_, a package that makes it easy to work with Bayesian GLMMs in Python, but it could be used independently as a backend for another library.


Dependencies
============
formulae requires a working Python interpreter (3.7+) and the libraries NumPy, SciPy and Pandas with versions specified in the (see `requirements.txt <https://github.com/bambinos/formulae/blob/master/requirements.txt>`_ for version information).

Installation
============
The latest release of formulae can be installed using pip:

.. code-block:: bash

   pip install formulae

Alternatively, if you want the development version of the package you can install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/bambinos/formulae.git

Contributors
============
See the `GitHub contributor page <https://github.com/bambinos/formulae/graphs/contributors>`_.

Contents
========

.. toctree::
   :maxdepth: 4

   api_reference

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
