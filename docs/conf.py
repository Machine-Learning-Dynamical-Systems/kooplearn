import os
import sys

sys.path.insert(0, os.path.abspath("../src/"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "kooplearn"
copyright = "2025, kooplearn team"
author = "kooplearn team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "nbsphinx",
    "myst_parser",
    "sphinx_iconify",
]

autosummary_generate = True
napoleon_use_ivar = True

source_suffix = {
    ".rst": "restructuredtext",
}

html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.png"


nbsphinx_epilog = """
.. footbibliography::
"""

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme_options = {
    "accent_color": "teal",
    "github_url": "https://github.com/Machine-Learning-Dynamical-Systems/kooplearn",
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
    ]
}

html_context = {
    "source_type": "github",
    "source_user": "Machine-Learning-Dynamical-Systems",
    "source_repo": "kooplearn",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
bibtex_bibfiles = ["bibliography.bib"]

html_theme = "shibuya"
html_static_path = ["_static"]
myst_enable_extensions = ["amsmath", "dollarmath", "html_image"]
nb_execution_mode = "off"

html_css_files = ["custom_rules.css"]
