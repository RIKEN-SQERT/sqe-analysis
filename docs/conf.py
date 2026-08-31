import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "sqe-analysis"
author = "The sqe-analysis contributors"
copyright = "%Y, The sqe-analysis contributors"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",

    # small extension that adds :autosummary: option to .. automodule::, so that we get summary tables similar to '.. autosummary::'
    "autodocsumm",

    "myst_nb",  # to enable {code-cell} - also activates myst_parser
]

# myst-nb creates new ipynb files under _build, which sphinx treats as new files
# so sphinx-autobuild goes into an infinite loop if we don't have this.
# see also https://myst-nb.readthedocs.io/en/latest/computation/execute.html,
# although the default seems fine for now.
exclude_patterns = [
    "_build"
]

suppress_warnings = [
    "mystnb.unknown_mime_type"  # "WARNING: skipping unknown output mime type: application/vnd.holoviews_load.v0+json"
]

# TODO: try different themes, good options are furo and pydata-sphinx-theme
#html_theme = "furo"

# TODO: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html for xarray
