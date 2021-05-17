# Contributing to TileDB-CF-Py

Hi! Thank you for your interest in contributing to TileDB-CF-Py. The following notes are intended to help you file issues, bug reports, or contribute code to this open source project.

## Contributing Checklist

* Reporting a bug?  Please read [how to file a bug report](#reporting-a-bug) section to make sure sufficient information is included.

* Contributing code? You rock! Be sure to [review the contributor section](#contributing-code) for helpful tips on the tools we use to build this project, format code, and issue pull requests (PR)'s.

Note: All participants in TileDB spaces are expected to adhere to a high standard of profectionalism in all interactions. See the [code of conduct](CODE_OF_CONDUCT.md) for more information.

## Reporting a Bug

A useful bug report filed as a GitHub issue provides information about how to reproduce the error.

1. Before opening a new [GitHub issue](https://github.com/TileDB-Inc/TileDB-CF-Py/issues) try searching the existing issues to see if someone else has already noticed the same problem.

2. When filing a bug report, provide where possible:

    * The version of TileDB-CF-Py or if a `dev` version, the specific commit that triggers the error.
    * The full error message, including the backtrace (if possible).
    * A minimal working example, i.e. the smallest chunk of code that triggers the error. Ideally, this should be code that can be a small reduced python file. If the code to reproduce is somewhat long, consider putting it in a [gist](https://gist.github.com).

3. When pasting code blocks or output, put triple backquotes (\`\`\`) around the text so GitHub will format it nicely. Code statements should be surrounded by single backquotes (\`). See [GitHub's guide on Markdown](https://guides.github.com/features/mastering-markdown) for more formatting tricks.

## Contributing Code

*By contributing code to TileDB-CF-Py, you are agreeing to release it under the [MIT License](https://github.com/TileDB-Inc/TileDB/tree/dev/LICENSE).*

### Quickstart Workflow

[From a fork of TileDB-CF-Py](https://help.github.com/articles/fork-a-repo/)

```bash
git clone https://github.com/username/TileDB-CF-Py
pip install -e '.[parallel]'
git checkout -b <my_initials>/<my_bugfix/feature_branch>
# ... code changes ...
./tools/lint.sh # run linters
tox # run all test suites with tox
git commit -a -m "descriptive commit message"
git push --set-upstream origin <my_initials>/<my_bugfix_branch>
```

[Issue a PR from your updated TileDB-CF-Py fork](https://help.github.com/articles/creating-a-pull-request-from-a-fork/)

Branch conventions:

* `dev` is the development branch of TileDB-CF-Py, all PR's are merged into `dev`.
* `release-x.y.z` are major / bugfix release branches.

### Building Locally for Development

This project uses setuptools for its build system, and can be built locally using pip. It is recommended you set-up a Python virtual environment with your preferred method before installing. Once the virtual environment is activated, install `tiledb.cf` as 'editable' using pip:

```bash
pip install -e .
```

The following tools are used for testing, linting, and formatting. You may want to install them either in the local virtual environment or as command line tools for you system:

* black
* flake8
* mypy
* pytest (with pytest-cov)
* tox


### Formatting, Style, and Linting

* 4 spaces per indentation level not tabs
* class names use `CamelCase`
* member functions, variables use `snake_case`
* private module or class member use a leading underscore `_local_variable`
* comments are good, the project uses Google-style docstrings with type hints
* format code using [black](https://pypi.org/project/black/) and [isort](https://pypi.org/project/isort/)
* lint code using [flake8](https://pypi.org/project/flake8/) and [mypy](https://pypi.org/project/mypy/)

It is highly recommended to run formatting and linting tools before every commit. This can be automated by activating the pre-commit hook `tools/hooks/pre-commit.sh`. To do this symlink or copy `tools/hooks/pre-commit.sh` to `.git/hooks/pre-commit` in the local directory. Note that the pre-commit hook may fail due to unstaged changes. You may wish to stash these changes before committing. This can be done as follows:

```bash
git add <files-to-be-added>
git stash --keep-index
git commit
git stash pop
```

### Testing

The testing for this project uses pytest and tox. Currently, tox is set-up to test Python versions 3.7, 3.8, and 3.9. This requires you to have `python3.7`, `python3.8`, and `python3.9` accessible to tox. Tests can be run by executing `tox`.

It is strongly recommended that you run the full tox test suite before submitting code for a pull request.

### Pull Requests

* `dev` is the development branch, all PR’s should be rebased on top of the latest `dev` commit.

* Commit changes to a local branch.  The convention is to use your initials to identify branches.  Branch names should be identifiable and reflect the feature or bug that they want to address / fix. This helps in deleting old branches later.

* Make sure the test suite passes by running `tox`.

* When ready to submit a PR, `git rebase` the branch on top of the latest `dev` commit.  Be sure to squash / cleanup the commit history so that the PR preferably one, or a couple commits at most.  Each atomic commit in a PR should be able to pass the test suite.

* Run the formatting (`isort`, `black`) and linting tools (`flake8`, `mypy`) before submitting a final PR. Make sure that your contribution generally follows the format and naming conventions used by surrounding code.

* Update the [HISTROY.md](HISTORY.md) with any changes/adds/removes to user-facing API or system behavior. Make sure to note any non-backward compatible changes as a breaking change.

* Submit a PR, writing a descriptive message.  If a PR closes an open issue, reference the issue in the PR message (e.g. If an issue closes issue number 10, you would write `closes #10`)

* Make sure CI (continuous integration) is passing for your PR.

### Resources

* TileDB-CF-Py
  * [Issues](https://github.com/TileDB-Inc/TileDB-CF-Py/issues)
  * [Documentation](https://docs.tiledb.com/geospatial)

* TileDB
  * [Homepage](https://tiledb.com)
  * [Documentation](https://docs.tiledb.com/main/)
  * [Forum](https://forum.tiledb.io/)
  * [Organization](https://github.com/TileDB-Inc/)

* Github / Git
  * [Git cheatsheet](https://services.github.com/on-demand/downloads/github-git-cheat-sheet/)
  * [Github Documentation](https://help.github.com/)
  * [Forking a Repo](https://help.github.com/articles/fork-a-repo/)
  * [More Learning Resources](https://help.github.com/articles/git-and-github-learning-resources/)
