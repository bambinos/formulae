Examples
========

The core function of this package is ``design_matrices()``. It returns
an object of class ``DesignMatrices`` that contains information about
the response, common effects and group specific effects.

.. code:: ipython3

    import pandas as pd
    import numpy as np
    
    from formulae import design_matrices

.. code:: ipython3

    np.random.seed(1234)

.. code:: ipython3

    SIZE = 20
    CNT = 20
    data = pd.DataFrame(
        {
            'x': np.random.normal(size=SIZE), 
            'y': np.random.normal(size=SIZE),
            'z': np.random.normal(size=SIZE),
            '$2#abc': np.random.normal(size=SIZE),
            'g1': np.random.choice(['a', 'b', 'c'], SIZE),
            'g2': np.random.choice(['YES', 'NO'], SIZE)
        }
    )

We can use both functions taht are loaded in the top environment as well
as non-syntactic names passed within \``. Specification of group
specific effects is much like what you have in R package ``lme4``.

.. code:: ipython3

    design = design_matrices("y ~ np.exp(x) + `$2#abc` + (z|g1)", data)

.. code:: ipython3

    print(design.response)
    print(design.response.design_vector)


.. parsed-literal::

    ResponseVector(name=y, type=numeric, length=20)
    [[-0.20264632]
     [-0.65596934]
     [ 0.19342138]
     [ 0.55343891]
     [ 1.31815155]
     [-0.46930528]
     [ 0.67555409]
     [-1.81702723]
     [-0.18310854]
     [ 1.05896919]
     [-0.39784023]
     [ 0.33743765]
     [ 1.04757857]
     [ 1.04593826]
     [ 0.86371729]
     [-0.12209157]
     [ 0.12471295]
     [-0.32279481]
     [ 0.84167471]
     [ 2.39096052]]


.. code:: ipython3

    print(design.common)
    print(design.common.design_matrix)


.. parsed-literal::

    CommonEffectsMatrix(
      shape: (20, 3),
      terms: {
        'Intercept': {type=Intercept, cols=slice(0, 1, None)},
        'np.exp(x)': {type=call, cols=slice(1, 2, None)},
        '$2#abc': {type=numeric, cols=slice(2, 3, None)}
      }
    )
    [[ 1.          1.6022921  -0.97423633]
     [ 1.          0.30392458 -0.07034488]
     [ 1.          4.19002612  0.30796886]
     [ 1.          0.73150451 -0.20849876]
     [ 1.          0.48646577  1.03380073]
     [ 1.          2.42823083 -2.40045363]
     [ 1.          2.36218825  2.03060362]
     [ 1.          0.52912874 -1.14263129]
     [ 1.          1.01582021  0.21188339]
     [ 1.          0.10617305  0.70472062]
     [ 1.          3.15830574 -0.78543521]
     [ 1.          2.69647677  0.46205974]
     [ 1.          2.59431919  0.70422823]
     [ 1.          0.13248911  0.52350797]
     [ 1.          0.71599839 -0.92625431]
     [ 1.          1.00212061  2.00784295]
     [ 1.          1.49998246  0.22696254]
     [ 1.          1.33521448 -1.15265911]
     [ 1.          3.74775949  0.63197945]
     [ 1.          0.21290578  0.03951269]]


.. code:: ipython3

    print(design.common['$2#abc'])


.. parsed-literal::

    [[-0.97423633]
     [-0.07034488]
     [ 0.30796886]
     [-0.20849876]
     [ 1.03380073]
     [-2.40045363]
     [ 2.03060362]
     [-1.14263129]
     [ 0.21188339]
     [ 0.70472062]
     [-0.78543521]
     [ 0.46205974]
     [ 0.70422823]
     [ 0.52350797]
     [-0.92625431]
     [ 2.00784295]
     [ 0.22696254]
     [-1.15265911]
     [ 0.63197945]
     [ 0.03951269]]


.. code:: ipython3

    print(design.group)
    print(design.group.design_matrix) # note it is a sparse matrix


.. parsed-literal::

    GroupEffectsMatrix(
      shape: (40, 6),
      terms: {
        '1|g1': {type=Intercept, idxs=(slice(0, 20, None), slice(0, 3, None))},
        'z|g1': {type=numeric, idxs=(slice(20, 40, None), slice(3, 6, None))}
      }
    )
      (0, 0)	1.0
      (1, 0)	1.0
      (7, 0)	1.0
      (9, 0)	1.0
      (13, 0)	1.0
      (14, 0)	1.0
      (16, 0)	1.0
      (17, 0)	1.0
      (18, 0)	1.0
      (2, 1)	1.0
      (3, 1)	1.0
      (6, 1)	1.0
      (10, 1)	1.0
      (11, 1)	1.0
      (12, 1)	1.0
      (15, 1)	1.0
      (19, 1)	1.0
      (4, 2)	1.0
      (5, 2)	1.0
      (8, 2)	1.0
      (20, 3)	0.07619958783723642
      (21, 3)	-0.5664459304649568
      (27, 3)	0.018289191349219306
      (29, 3)	0.2152685809694434
      (33, 3)	-0.10091819994891389
      (34, 3)	-0.5482424491868549
      (36, 3)	0.3540203321992379
      (37, 3)	-0.0355130252781402
      (38, 3)	0.5657383060625951
      (22, 4)	0.036141936684072715
      (23, 4)	-2.0749776006900293
      (26, 4)	-0.1367948332613474
      (30, 4)	0.841008794931391
      (31, 4)	-1.4458100770443063
      (32, 4)	-1.4019732815008439
      (35, 4)	-0.14461950836938436
      (39, 4)	1.5456588046255575
      (24, 5)	0.24779219974854666
      (25, 5)	-0.8971567844396987
      (28, 5)	0.7554139823981354


:math:`Z` matrix can be subsetted by passing the name of the group
specific term.

.. code:: ipython3

    print(design.group['z|g1'])


.. parsed-literal::

    [[ 0.07619959  0.          0.        ]
     [-0.56644593  0.          0.        ]
     [ 0.          0.03614194  0.        ]
     [ 0.         -2.0749776   0.        ]
     [ 0.          0.          0.2477922 ]
     [ 0.          0.         -0.89715678]
     [ 0.         -0.13679483  0.        ]
     [ 0.01828919  0.          0.        ]
     [ 0.          0.          0.75541398]
     [ 0.21526858  0.          0.        ]
     [ 0.          0.84100879  0.        ]
     [ 0.         -1.44581008  0.        ]
     [ 0.         -1.40197328  0.        ]
     [-0.1009182   0.          0.        ]
     [-0.54824245  0.          0.        ]
     [ 0.         -0.14461951  0.        ]
     [ 0.35402033  0.          0.        ]
     [-0.03551303  0.          0.        ]
     [ 0.56573831  0.          0.        ]
     [ 0.          1.5456588   0.        ]]


Reference class example
-----------------------

This feature is taken from current Bambi behavior (you don’t find it in
Patsy or formulaic)

.. code:: ipython3

    design = design_matrices('g2[YES] ~ x', data)
    print(design.response)
    print(design.response.design_vector)


.. parsed-literal::

    ResponseVector(name=g2, type=categoric, length=20, refclass=YES)
    [[0]
     [1]
     [1]
     [0]
     [0]
     [0]
     [1]
     [0]
     [0]
     [0]
     [0]
     [0]
     [0]
     [1]
     [1]
     [0]
     [1]
     [0]
     [0]
     [1]]

