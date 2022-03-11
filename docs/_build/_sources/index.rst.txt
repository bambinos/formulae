.. image:: logo/formulae_large.png
   :alt: formulae
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

This package has been written to make it easier to specify models with group effects in `Bambi <https://github.com/bambinos/bambi>`_,
a package that makes it easy to work with Bayesian GLMMs in Python, but it could be used
independently as a backend for another library.
The approach in this library is to extend classical statistical formulas in a similar way than in
R package `lme4 <https://CRAN.R-project.org/package=lme4>`_.


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
While Patsy is great, flexible, and solid, it does not support formulas with mixed model effects.
At that time, Bambi developers would ask users for random effects to be passed in a list separated
from the model formula which was cumbersome for models with several terms.

It could have been possible to attempt to modify Patsy to make it work with mixed effects formulas.
But lack of familiarity with the internals of the library and the motivation to write something
completely custom to our needs predominated and formulae development started.

From the very beginning, formulae was built with Bambi needs in mind. That's why its main function,
``design_matrices()``, does not return an object that can be directly used as a design matrix, but
as a wrapper for classes containing the design matrices as well as useful methods and attributes.
These methods and attributes are extensively used within Bambi to build internal objects that give
shape to Bambi models.

formulae was officially incorporated into Bambi a couple of months after its inception. Several
updates, bug fixes, and improvements took place from that moment. While there is still much work to
be done, the current shape of formulae does at good job at meeting needs in Bambi.

Future efforts are more likely to be concentrated around adding new features and making the library
more solid in general, instead of converting formulae into a high-level library that can be used as
a direct replacement of Patsy.

But formulae couldn't have existed if it wasn't for the following projects that served as both
inspiration and source of information

* `The work where everything started <https://doi.org/10.2307/2346786>`_: Wilkinson, G., & Rogers, C. Symbolic Description of Factorial Models for Analysis of Variance. Journal of the Royal Statistics Society 22, pp. 392–399, 1973.
* `R <https://www.r-project.org/>`_: Probably the most popular implementation of Wilkinson's formulas.
* `lme4 <https://CRAN.R-project.org/package=lme4>`_: For the ``|`` operator to extend Wilkinson's formulas to mixed effects models and helpful information on how to compute mixed effects matrices.
* `Patsy <https://patsy.readthedocs.io/>`_: The most widely used implementation of Wilkinson's formulas in Python. Its implementation helped us to write formulae, especially its module and documentation on evaluation environments.
* `Formulaic <https://matthewwardrop.github.io/formulaic/>`_: Another implementation of Wilkinson's formulas in Python that we came across in the middle of this journey. The usage of the backtick operator ````` and the quote operator ``{`` are taken from this library.


Finally, if you came here because you only need to obtain design matrices for linear models with
fixed effects, you'd better use Patsy or Formulaic. They are much more friendly and go straight to
the point of returning a design matrix. On the contrary, if you are a developer or someone who needs
to automatically generate design matrices for mixed-effects models, have a try with formulae and
feel free to reach out to us if you have any question or sugerence.

Contributors
============
See the `GitHub contributor page <https://github.com/bambinos/formulae/graphs/contributors>`_.

Contents
========

.. toctree::
   :maxdepth: 4

   notebooks/getting_started
   api_reference
   internals

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
