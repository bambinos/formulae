Internals
**********

This reference provides detailed documentation for modules and classes that are important to
developers who want to include formulae in their library.


.. toctree::
  :maxdepth: 4


:mod:`matrices`
================================

These objects are not intended to be used by end users. But developers working with formulae will
need some familiarity with them, especially if you want to take advantage of features like obtaining
a design matrix from an existing design but evaluated with new data.

.. autoclass:: formulae.matrices.ResponseVector
  :members:
  :private-members:
  :special-members: __getitem__

.. autoclass:: formulae.matrices.CommonEffectsMatrix
  :members:
  :private-members:
  :special-members: __getitem__

.. autoclass:: formulae.matrices.GroupEffectsMatrix
  :members:
  :private-members:
  :special-members: __getitem__


:mod:`terms`
================================

These are internal components of the model that are not expected to be used by end users.
Developers won't (normally) need to access these objects either.
But reading this documentation may help you understand how formulae works, with both its advantages
and disadvantages.

.. autoclass:: formulae.terms.Variable
  :members:
  :private-members:

.. autoclass:: formulae.terms.Call
  :members:
  :private-members:

.. autoclass:: formulae.terms.Term
  :members:
  :private-members:
  :special-members: __add__, __sub__, __mul__, __matmul__, __truediv__, __or__, __pow__

.. autoclass:: formulae.terms.Intercept
  :members:
  :private-members:
  :special-members: __add__, __sub__, __or__
