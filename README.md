<img src="docs/logo/formulae_large.png" width=250></img>

[![PyPI version](https://badge.fury.io/py/formulae.svg)](https://badge.fury.io/py/formulae)
[![codecov](https://codecov.io/gh/bambinos/formulae/branch/master/graph/badge.svg)](https://codecov.io/gh/bambinos/formulae)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


# formulae

formulae is a Python library that implements Wilkinson's formulas for mixed-effects models. The main difference with other implementations like [Patsy](https://github.com/pydata/patsy) or [formulaic](https://github.com/matthewwardrop/formulaic) is that formulae can work with formulas describing a model with both common and group specific effects (a.k.a. fixed and random effects, respectively).

This package has been written to make it easier to specify models with group effects in [Bambi](https://github.com/bambinos/bambi), a package that makes it easy to work with Bayesian GLMMs in Python, but it could be used independently as a backend for another library. The approach in this library is to extend classical statistical formulas in a similar way than in R package [lme4](https://CRAN.R-project.org/package=lme4).

**Note:** While this package is working, there is no online documentation yet and you may find bugs within the code. You are encouraged to play with this library and give feedback about it, but it is not recommended to incorporate formulae in a larger project at this early stage of development.

## Installation

formulae requires a working Python interpreter (3.7+) and the libraries numpy, scipy and pandas with versions specified in the [requirements.txt](https://github.com/bambinos/formulae/blob/master/requirements.txt) file.

Assuming a standard Python environment is installed on your machine (including pip), the latest release of formulae can be installed in one line using pip:

`pip install formulae`

Alternatively, if you want the development version of the package you can install from GitHub:

`pip install git+https://github.com/bambinos/formulae.git`

## Example code

The main function in this library is `design_matrices()`. It takes a formula and a pandas DataFrame and returns an object of class `DesignMatrices` that contains information about the response, the common effects, and the group specific effects that can be accessed with the attributes `.response`, `.common`, and `.group` respectively.


```python
import numpy as np
import pandas as pd

from formulae import design_matrices
```


```python
np.random.seed(1234)
df = pd.DataFrame({
    'y_num': np.random.normal(size=10),
    'y_cat': np.random.choice(['A', 'B'], size=10),
    'x': np.random.normal(size=10),
    'g': np.random.choice(['Group 1', 'Group 2', 'Group 3'], size=10)
})
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_num</th>
      <th>y_cat</th>
      <th>x</th>
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.471435</td>
      <td>B</td>
      <td>-0.304260</td>
      <td>Group 1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.190976</td>
      <td>A</td>
      <td>0.861661</td>
      <td>Group 3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.432707</td>
      <td>B</td>
      <td>-0.689927</td>
      <td>Group 3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.312652</td>
      <td>B</td>
      <td>0.187497</td>
      <td>Group 1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.720589</td>
      <td>A</td>
      <td>0.604309</td>
      <td>Group 2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.887163</td>
      <td>A</td>
      <td>-0.183014</td>
      <td>Group 2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.859588</td>
      <td>B</td>
      <td>-1.126502</td>
      <td>Group 1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-0.636524</td>
      <td>A</td>
      <td>1.658873</td>
      <td>Group 1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.015696</td>
      <td>A</td>
      <td>-0.660441</td>
      <td>Group 1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-2.242685</td>
      <td>B</td>
      <td>1.041086</td>
      <td>Group 2</td>
    </tr>
  </tbody>
</table>
</div>




### Example 1

A simple linear model with numeric response, numeric common effects and varying slope and intercept for each level of `g`.


```python
design = design_matrices("y_num ~ x + (x|g)", df)
```


```python
print(design.response)
print(design.response.design_vector)
```

    ResponseVector(name=y_num, type=numeric, length=10)
    [[ 0.47143516]
     [-1.19097569]
     [ 1.43270697]
     [-0.3126519 ]
     [-0.72058873]
     [ 0.88716294]
     [ 0.85958841]
     [-0.6365235 ]
     [ 0.01569637]
     [-2.24268495]]



```python
print(design.common)
print(design.common.design_matrix) # this can be printed as a pandas.DataFrame with design.common.as_dataframe()
```

    CommonEffectsMatrix(
      shape: (10, 2),
      terms: {
        'Intercept': {type=intercept, cols=slice(0, 1, None), full_names=['Intercept']},
        'x': {type=numeric, cols=slice(1, 2, None), full_names=['x']}
      }
    )
    [[ 1.         -0.30426018]
     [ 1.          0.861661  ]
     [ 1.         -0.68992667]
     [ 1.          0.18749737]
     [ 1.          0.60430874]
     [ 1.         -0.18301422]
     [ 1.         -1.12650247]
     [ 1.          1.65887284]
     [ 1.         -0.66044141]
     [ 1.          1.04108597]]


Before exploring the group level effects it should be noted that formulae returns a sparse matrix in CSC format. If it is the case the matrix is not that big and you want to print it as any other matrix, you can call `design.group.design_matrix.toarray()`


```python
print(design.group)
print(design.group.design_matrix.toarray())
```

    GroupEffectsMatrix(
      shape: (20, 6),
      terms: {
        '1|g': {type=intercept, groups=['Group 1', 'Group 2', 'Group 3'], idxs=(slice(0, 10, None), slice(0, 3, None)), full_names=['1|g[Group 1]', '1|g[Group 2]', '1|g[Group 3]']},
        'x|g': {type=numeric, groups=['Group 1', 'Group 2', 'Group 3'], idxs=(slice(10, 20, None), slice(3, 6, None)), full_names=['x|g[Group 1]', 'x|g[Group 2]', 'x|g[Group 3]']}
      }
    )
    [[ 1.          0.          0.          0.          0.          0.        ]
     [ 0.          0.          1.          0.          0.          0.        ]
     [ 0.          0.          1.          0.          0.          0.        ]
     [ 1.          0.          0.          0.          0.          0.        ]
     [ 0.          1.          0.          0.          0.          0.        ]
     [ 0.          1.          0.          0.          0.          0.        ]
     [ 1.          0.          0.          0.          0.          0.        ]
     [ 1.          0.          0.          0.          0.          0.        ]
     [ 1.          0.          0.          0.          0.          0.        ]
     [ 0.          1.          0.          0.          0.          0.        ]
     [ 0.          0.          0.         -0.30426018  0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.861661  ]
     [ 0.          0.          0.          0.          0.         -0.68992667]
     [ 0.          0.          0.          0.18749737  0.          0.        ]
     [ 0.          0.          0.          0.          0.60430874  0.        ]
     [ 0.          0.          0.          0.         -0.18301422  0.        ]
     [ 0.          0.          0.         -1.12650247  0.          0.        ]
     [ 0.          0.          0.          1.65887284  0.          0.        ]
     [ 0.          0.          0.         -0.66044141  0.          0.        ]
     [ 0.          0.          0.          0.          1.04108597  0.        ]]


But if you are interested only in the sub-matrix corresponding to a given group specific effect, you can use `design.group['level_name']` as follows


```python
design.group['x|g']
```




    array([[-0.30426018,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.861661  ],
           [ 0.        ,  0.        , -0.68992667],
           [ 0.18749737,  0.        ,  0.        ],
           [ 0.        ,  0.60430874,  0.        ],
           [ 0.        , -0.18301422,  0.        ],
           [-1.12650247,  0.        ,  0.        ],
           [ 1.65887284,  0.        ,  0.        ],
           [-0.66044141,  0.        ,  0.        ],
           [ 0.        ,  1.04108597,  0.        ]])



### Example 2

A categorical response and a linear predictor that has an interaction between a categorical variable and a function call. In this example we use the `variable['level']` notation that is taken from the original version of [Bambi](https://github.com/bambinos/bambi) and makes it easier to indicate which level represents a success in a categorical response.


```python
design = design_matrices("y_cat['A'] ~ np.exp(x) * g", df)
```


```python
print(design.response)
print(design.response.design_vector)
```

    ResponseVector(name=y_cat, type=categoric, length=10, refclass=A)
    [[0]
     [1]
     [0]
     [0]
     [1]
     [1]
     [0]
     [1]
     [1]
     [0]]



```python
design.common
```




    CommonEffectsMatrix(
      shape: (10, 6),
      terms: {
        'Intercept': {type=intercept, cols=slice(0, 1, None), full_names=['Intercept']},
        'np.exp(x)': {type=numeric, cols=slice(1, 2, None), full_names=['np.exp(x)']},
        'g': {type=categoric, levels=['Group 1', 'Group 2', 'Group 3'], reference=Group 1, encoding=reduced, cols=slice(2, 4, None), full_names=['g[Group 2]', 'g[Group 3]']},
        'np.exp(x):g': {type=interaction, cols=slice(4, 6, None), full_names=['np.exp(x):g[Group 2]', 'np.exp(x):g[Group 3]'],
          vars={
            np.exp(x): {type=numeric},
            g: {type=categoric, levels=['Group 1', 'Group 2', 'Group 3'], reference=Group 1, encoding=reduced}
        }}
      }
    )




```python
design.common.as_dataframe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Intercept</th>
      <th>np.exp(x)</th>
      <th>g[Group 2]</th>
      <th>g[Group 3]</th>
      <th>np.exp(x):g[Group 2]</th>
      <th>np.exp(x):g[Group 3]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.737669</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2.367089</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>2.367089</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.501613</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.501613</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.206227</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.829987</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.829987</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.832756</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.832756</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>0.324165</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.0</td>
      <td>5.253386</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>0.516623</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>2.832291</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.832291</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Notes

* The `data` argument only accepts objects of class `pandas.DataFrame`.
* `y ~ .` is not implemented and won't be implemented in a first version. However, it is planned to be included in the future.
