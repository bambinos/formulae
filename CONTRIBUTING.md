# Guidelines for Contributing

As a scientific community-driven software project, Formulae welcomes contributions
from interested individuals and groups. These guidelines help potential contributors
make their work consistent with the project and maximize the probability that it can
be reviewed and merged efficiently.

There are four main ways to contribute to Formulae, in descending order of scope:

- Add or improve functionality in the codebase.
- Fix open issues, from low-level bugs to higher-level design problems.
- Improve the documentation (`docs`) or examples in the documentation notebooks.
- Report bugs or request enhancements.

## Opening issues

We appreciate reports of problems with Formulae. Please file them in the
[GitHub issue tracker](https://github.com/bambinos/formulae/issues), rather than on
social media or by direct email to the developers.

Before opening an issue, search existing issues and pull requests for relevant
keywords to check whether the problem is already being addressed.

## Contributing code via pull requests

Although issue reports are valuable, we strongly encourage contributors to submit
patches through pull requests—especially for small fixes such as documentation or
typographical corrections. New functionality is welcome through pull requests too.

The preferred workflow is to fork the
[Formulae repository](https://github.com/bambinos/formulae), clone the fork locally,
and develop on a feature branch.

For the final checks, see the [pull request checklist](#pull-request-checklist).

### Code formatting

Formulae follows PEP 8 and uses [Black](https://black.readthedocs.io/) to format
Python code. The project's pre-commit hooks enforce the configured formatting and
basic file checks.

### Docstring formatting

Docstrings follow the
[NumPy documentation style](https://numpydoc.readthedocs.io/en/latest/format.html).
Document additions and meaningful changes to the codebase; when in doubt, add a
docstring.

## Steps

1. Fork the [project repository](https://github.com/bambinos/formulae) with the
   **Fork** button near the top right of the repository page. This creates a copy
   under your GitHub account.

2. Clone your fork and add the base repository as an upstream remote:

   ```bash
   git clone git@github.com:<your-github-handle>/formulae.git
   cd formulae
   git remote add upstream git@github.com:bambinos/formulae.git
   ```

3. Create a feature branch for your work:

   ```bash
   git checkout -b <your-branch-name>
   ```

   Always work on a feature branch rather than routinely working on `master`.

4. Set up the [Pixi](https://pixi.sh/) development environment (only once):

   1. If Pixi is not installed, follow its
      [official installation instructions](https://pixi.sh/latest/installation/).
   2. Install the `dev` environment, which includes the dependencies used for
      development, testing, linting, and documentation:

      ```bash
      pixi install -e dev
      ```

   3. Install the pre-commit hooks:

      ```bash
      pixi run -e dev pre-commit-setup
      ```

5. Activate the environment when you want an interactive development shell:

   ```bash
   pixi shell -e dev
   ```

   Use `exit` to leave the shell. You can also run individual commands without
   activating it, for example `pixi run -e dev pytest tests`.

6. Develop the feature on your branch. Stage and commit your changes locally:

   ```bash
   git add <modified-files>
   git commit -m "Summarize the change"
   ```

   Before publishing your branch, synchronize it with the base repository:

   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

   Then push the branch to your fork:

   ```bash
   git push -u origin <your-branch-name>
   ```

7. Open the GitHub page for your fork and click **Pull Request** to submit your
   changes for review.

## Building the documentation locally

Formulae's documentation is built with [Quarto](https://quarto.org/) and
[Quartodoc](https://machow.github.io/quartodoc/) for the API reference.
The Pixi `dev` environment includes the required Python dependencies.

### Prerequisites

- The `dev` Pixi environment is installed (see the [setup steps](#steps)).
- Install Quarto separately by following its
  [official installation instructions](https://quarto.org/docs/get-started/).

### Building the docs

From the repository root, first build the API reference and its interlinks:

```bash
cd docs
pixi run -e dev python -m quartodoc build --verbose
pixi run -e dev python -m quartodoc interlinks
```

Then render the website:

```bash
quarto render
```

The generated site is available at `docs/_site/index.html`.
Quarto does not execute the documentation notebook while rendering it.

## Pull request checklist

Before submitting a pull request, please check the following:

- If the pull request addresses an issue, describe it in the title and reference
  the issue number in the description.
- Use a draft pull request for incomplete work, early API feedback, or collaboration.
- Add or update tests for new functionality and run relevant documentation notebooks
  when your changes affect them.
- Public functions and methods should have informative NumPy-style docstrings.
- Update documentation when a user-visible API or behavior changes.
- Run the configured pre-commit hooks:

  ```bash
  pixi run -e dev pre-commit run --all
  ```

- Run Pylint:

  ```bash
  pixi run -e dev pylint formulae
  ```

- Run the test suite:

  ```bash
  pixi run -e dev pytest tests
  ```

This guide was adapted from the
[Bambi contributing guide](https://github.com/bambinos/bambi/blob/main/CONTRIBUTING.md).
