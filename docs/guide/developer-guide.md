# Developer guide

## Basic development flow

- Run tests: `uv run pytest` in the repository root
- Checks:
    - `uv run ruff check`
    - `uv run ruff format`

    These can be run on every commit using e.g. [prek](https://prek.j178.dev/), but it is not strictly required.

## Documentation

There are two kinds of documentation:
- API documentation, which lives in python source code docstrings. These are written following the [Google docstring convention](https://google.github.io/styleguide/pyguide.html) and rendered by [Sphinx](https://www.sphinx-doc.org/) with the [napoleon extension](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- Stand-alone documentation, like this guide. This kind of documentation mixes prose and code, as well as the output from the code such as plots, in a style similar to Jupyter notebooks. This documentation is authored in [MyST](https://mystmd.org/), which is an extension of [Markdown](https://commonmark.org/), and it is rendered by Sphinx using the [myst-parser](https://myst-parser.readthedocs.io/en/stable/) extension.

The documentation can be built by running the command `make docs` in the repository root. To clean up the generated documentation, use `make docs-clean`. To automatically regenerate and reload the docs when they are edited, use `make docs-live` and open the URL shown by the command.

The API documentation is semi-automatically generated: There is a hand-written markdown file for each module in the `api/` folder. These files contain [autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) directives that render API documentation from the Python source code. Additionally, a table summarizing the contents of each module is generated using the third-party [autodocsumm](https://autodocsumm.readthedocs.io/en/latest/) extension. Here are some alternative approaches that were considered for generating the API documentation but were found to be insufficient:

- Using the [autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html) extension of Sphinx with the `:recursive:` option

    This automatically adds a table summarizing module contents to the top of each page, but doesn't actually generate the module documentation itself. This can be worked around by [tweaking the templates](https://github.com/sphinx-doc/sphinx/issues/7912) used by autosummary (see also [this StackOverflow answer](https://stackoverflow.com/questions/2701998/automatically-document-all-modules-recursively-with-sphinx-autodoc/62613202#62613202)), but this is hacky and a bit backwards: autosummary is a directive for generating a summary table, it should not be responsible for generating API documentation. See [this github issue](https://github.com/sphinx-doc/sphinx/issues/6829) for some further discussion.

- Using the [apidoc extension](https://www.sphinx-doc.org/en/master/usage/extensions/apidoc.html) or [CLI tool](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html) built into Sphinx

    This automatically generates a `.rst` file for each module, similarly to the hand-written markdown files. The downside is that the output is difficult to customize: it generates a deeply nested table of contents by default and there is no easy way to add an autosummary-style table of contents to the top of each module page. There is no easy way to use a custom template with apidoc. The currently implemented solution is effectively equivalent to running the `sphinx-apidoc` command once and treating the output as manually version-controlled source code.

- Using the third-party [AutoAPI extension](https://sphinx-autoapi.readthedocs.io/en/latest/index.html)
    
    The AutoAPI extension is actively maintained by the Read the Docs team, and it gets pretty close to what we want. It completely replaces / reimplements autodoc and autosummary. However, the table of contents it generates has the wrong structure, which requires custom templating to fix. It is also slow, generating the API documentation takes several seconds compared to less than 1 second for autodoc with the hand-written templates.

- Using the third-party [autodoc2](https://sphinx-autodoc2.readthedocs.io/en/latest/) extension

    autodoc2 is also close to what we want to do. It has the added bonus that it can use MyST markdown instead of RST. However, as of August 2026 it has not received updates in over two years, so it was not very carefully evaluated.

The semi-automatic approach was chosen as a compromise between the tradeoffs of the above options. The list of modules is not going to change often, so it is acceptable to write out by hand, in exchange for finer control over the table of contents and the layout of each module page, and with minimal dependence on third-party packages.


## LLM policy

TODO
