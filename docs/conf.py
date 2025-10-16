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
    "myst_nb",
    "sphinx_iconify",
]

autosummary_generate = True
napoleon_use_ivar = True

source_suffix = {
    ".rst": "restructuredtext",
    ".myst": "myst-nb",
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# pygments_style = "tango"
# pygments_style_dark = "material"

html_theme_options = {
    "accent_color": "teal",
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
bibtex_bibfiles = ["bibliography.bib"]

html_theme = "shibuya"
html_static_path = ["_static"]
myst_enable_extensions = ["amsmath", "dollarmath", "html_image"]
nb_execution_mode = "off"

html_css_files = ["custom_rules.css"]
