# Developer guide

## Documentation

There are two kinds of documentation:
- API documentation, which lives in python source code docstrings. These are written following the [Google docstring convention](https://google.github.io/styleguide/pyguide.html) and rendered by [Sphinx](https://www.sphinx-doc.org/) with the [napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- Stand-alone documentation, like this guide. This kind of documentation mixes prose and code, as well as the output from the code such as plots, in a style similar to Jupyter notebooks. This documentation is authored in [MyST](https://mystmd.org/), which is an extension of [Markdown](https://commonmark.org/), and it is rendered by Sphinx using the [myst-parser](https://myst-parser.readthedocs.io/en/stable/) extension.

The documentation can be built by running the command `make docs` in the repository root.

## LLM policy

TODO
