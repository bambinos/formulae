# Change Log

## v0.3.0 Unreleased

### New features

- We can create our own encodings such as Treatment or Sum encoding. These are subclasses of `Encoding`.
- Added two aliases `T` and `S` that are shorthands of `C(var, Treatment)` and `C(var, Sum)` respectively.

### Maintenance and fixes

* Fixed a bug in the addition of lower order terms when the original higher order term wasn't full-rank.
* Columns for categorical terms are integer by default. They are casted to float only if there are other float-valued columns in the design matrix.

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
