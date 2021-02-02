# formulae

**UNDER DEVELOPMENT**

The formula language is fully implemented (with a lot of bugs of course!) and you can generate
model matrices for models that only contain common effects (a.k.a. fixed effects).

You can check existing features in `Features.ipynb`.

## Misc notes

* `y ~ .` is not implemented and won't be implemented in a first version. However, it is planned
to include it in the future.

## WIP

The following are things that are not available yet, but are a work in progress.

- [ ] Implement group specific effects matrix
- [ ] Incorporate built-in functions like `center(x)`, `standardize(x)`, `C(x)`, etc.
- [ ] Write language syntax
- [ ] Create some basic documentation
- [ ] Add tests

