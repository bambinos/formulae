.. image:: logo/formulae_large.png
  :width: 250

===================================================

|PyPI version|
|Coverage|
|Black|

.. |PyPI version| image:: https://badge.fury.io/py/formulae.svg
   :target: https://badge.fury.io/py/formulae

.. |Coverage| image:: https://codecov.io/gh/bambinos/formulae/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/bambinos/formulae

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/ambv/black



formulae is a Python library that implements Wilkinson's formulas for mixed-effects models.

This package has been written to make it easier to specify models with group effects in `Bambi <https://github.com/bambinos/bambi>`_, a package that makes it easy to work with Bayesian GLMMs in Python, but it could be used independently as a backend for another library. The approach in this library is to extend classical statistical formulas in a similar way than in R package `lme4 <https://CRAN.R-project.org/package=lme4>`_.


Dependencies
============
formulae requires a working Python interpreter (3.7+) and the libraries NumPy, SciPy and pandas with
versions specified in the (see
`requirements.txt <https://github.com/bambinos/formulae/blob/master/requirements.txt>`_ for version
information).

Installation
============
The latest release of formulae can be installed using pip:

.. code-block:: bash

   pip install formulae

Alternatively, if you want the development version of the package you can install from GitHub:

.. code-block:: bash

   pip install git+https://github.com/bambinos/formulae.git

History, related projects and credits
======================================

formulae was built specifically to satisfy the need for a more concise manner to specify mixed
effects models in Bambi. Before formulae, Bambi used to rely on
`Patsy <https://patsy.readthedocs.io/>`_ to parse model formulas and build design matrices.
While Patsy is great and solid, it does not support formulas with mixed model effects. At that time,
Bambi developers decided to ask users for random effects to be passed in a list separated from the
model formula.

It could have been possible to attempt to modify Patsy to make it work with mixed effects formulas.
But lack of familiarity with the internals of the library and the motivation to write something
completely custom to our needs predominated and formulae development started.

From the very beginning, formulae was built with Bambi needs in mind. That's why its main function,
``design_matrices()``, does not return an object that can be directly used as a design matrix, but
as a wrapper for classes containing the design matrices as well as useful methods and attributes.
These methods and attributes are extensively used within Bambi to build internal objects that shape
a model.

formulae was officially incorporated into Bambi a couple of months after its inception. Several
updates, bug fixes, and improvements took place from that moment. Currently formulae works as
expected and no API changes are expected (functions may be added, but not removed).

Need to mention related projects and what we took from each of them.

There is of course more work to be done. formulae is not close to be as solid and flexible as Patsy
or even formulaic.

Mention a couple of things we would like to add.


Contributors
============
See the `GitHub contributor page <https://github.com/bambinos/formulae/graphs/contributors>`_.

Contents
========

.. toctree::
   :maxdepth: 4

   api_reference
   internals

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
