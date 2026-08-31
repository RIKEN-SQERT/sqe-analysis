import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "sqe-analysis"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",

    # small extension that adds :autosummary: option to .. automodule::, so that we get summary tables similar to '.. autosummary::'
    "autodocsumm",

    "myst_nb",  # to enable {code-cell} - also activates myst_parser
]

# TODO: try different themes, good options are furo and pydata-sphinx-theme
#html_theme = "furo"

# TODO: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html for xarray
