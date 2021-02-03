# formulae

The formula language is fully implemented (with a lot of bugs of course!) and you can generate
model matrices for models with both common and group specific effects
(a.k.a. fixed and random effects, respectively)

If you want to see some quick examples in action, see the [Examples]('https://github.com/bambinos/formulae/blob/master/examples/Examples.ipynb').
To check how a formula is parsed into a model description object, see [Features]('https://github.com/bambinos/formulae/blob/master/examples/Features.ipynb').

## Misc notes

* `y ~ .` is not implemented and won't be implemented in a first version. However, it is planned
to include it in the future.

## WIP

The following are things that are not available yet, but are a work in progress.

- [ ] Incorporate built-in functions like `center(x)`, `standardize(x)`, `C(x)`, etc.
- [ ] Write language syntax
- [ ] Create some basic documentation
- [ ] Add tests

