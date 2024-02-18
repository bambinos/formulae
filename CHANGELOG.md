# Changelog

## v0.X.X Unreleased

### New features

### Maintenance and fixes

### Documentation

### Deprecation

## v0.5.2

### Maintenance and fixes

- Update is_categorical_dtype for pandas >= 2.10 (#105)
- Add automatic versioning to the library (#106)
- Interpret True, False, and None as Python literals (#107)

## v0.5.1

### Maintenance and fixes

- Fix bug when intercept is inserted after categorical variable (#102)

## v0.5.0

### New features

- The library now supports configuration variables (#95)
- Allow evaluations on new data with new categories (#96)
- Return ready to be used group-specific effects design matrix when there are new groups (#100)

### Maintenance and fixes

- Fix typo in __str__ (#99)

### Documentation

### Deprecation

## v0.4.0

### New features

- Implement a @register_stateful_transform decorator to register new stateful transformations (#85)

### Maintenance and fixes

- Fix evaluation of new offset terms with call arguments (#81)
- Multiple calls with unnamed attributes resulted in errors due to trying to hash unhashable types (#88)
- Term names with call components now use the proper lexeme when one argument is a string (#89)
- Detect stateful transformations by attribute instead of name (#90)
- Moved tests out of the `formulae` directory (#91)
- Use pyproject and improve deployment workflow (#92)

### Documentation

### Deprecation

## v0.3.4

### Maintenance and fixes

- Fixed a bug in the levels of interaction terms involving numeric terms with multiple columns (b4a1f73)

## v0.3.3

### Maintenance and fixes

- Fixed a bug in `CategoricalBox`. Now it considers the order of the categories if `data` is ordered and `levels` is `None` (#73)

## v0.3.2

### Maintenance and fixes

- Fixed a bug in `CategoricalBox` because it failed to convert categorical series to numpy arrays. Now it works. (#72)

## v0.3.1

### Maintenance and fixes

- Renamed `ResponseVector` to `ResponseMatrix` (#71)
- Renamed `design_vector` to `design_matrix` in `ResponseMatrix`(#71)
- Updated docstrings in `formulae/matrices.py` (#71)

### Deprecation

- Removed `binary` and `success` attributes from `ResponseMatrix` (#71)

## v0.3.0

### New features

- We can create our own encodings such as Treatment or Sum encoding. These are subclasses of `Encoding`.
- Added two aliases `T` and `S` that are shorthands of `C(var, Treatment)` and `C(var, Sum)` respectively.
- DesignVector, CommonEffectsMatrix and GroupEffectsMatrix now retrieve their values when passed to `np.array()` and `np.asarray()`.
- Add `poly()` stateful transform.
- `na_action` in `design_matrices()` can be `"pass"` which means not to drop or raise error about missing values. It just keeps them (#69)
- `design_matrices()` gains a new argument `extra_namespace`. It allows us to pass a dictionary of transformations that are made available in the environment where the formula is evaluated (#70)

### Maintenance and fixes

- Fixed a bug in the addition of lower order terms when the original higher order term wasn't full-rank.
- Columns for categorical terms are integer by default. They are casted to float only if there are other float-valued columns in the design matrix.
- Updated __str__ and __repr__ methods of `ResponseVector`, `CommonEffectsMatrix`, and `GroupEffectsMatrix`.
- Added __str__ and __repr__ methods for `DesignMatrices`.
- Added __get_item__ method for `DesignMatrices`.
- Added support for comparison operators within function calls.

## v0.2.0

- Added `bs()`, a stateful transformation for basis splines (#52)
- Subset notation in response terms can now be an identifier. It is interpreted as a string (#52)
- `True`, `False` and `None` are correctly evaluated in function calls arguments now (#53)
- The `.set_type()` on each term used to be called twice. Now we call it once (#53)
- Added the function `get_function_from_module()`. Now we don't use Python's `eval()` anymore (#53)

## v0.1.4

- Revert changes back to v0.1.2

## v0.1.3

- Revert changes back to v0.1.1

## v0.1.2

- Added `prop()` function to handle response terms that a proportion computed from two variables (#40)
- Categorical responses can contain more than one level (#40)
- Added `binary()` function to convert numeric/categorical variables into 0-1 variables (#41)
- Modified `C()` to make it simpler to obtain categorical variables and manipulate their levels (#41)
- Added `offset()` (#42)

## v0.1.1

### Maintenance and fixes

- Fix group-specific effects matrix shape (#35)

### Documentation

- Add a Getting started section to webpage and remove that content from README.md (#36)

## v0.1.0

### New features

- Factor of group specific effect can be an interaction now (#31)

### Maintenance and fixes

- Full interaction does not result in shared components anymore (#30)
- Add information about levels to interaction terms (#33)

### Deprecation

### Documentation

- Added **Internals** section to documentation (#28)

## v0.0.10

### New features

### Maintenance and fixes

- Nested stateful transofrmations now work as expected (#19)

### Deprecation

### Documentation

- Added a changelog file to track changes (#20)
- Formulae now has a github pages website (#21)
