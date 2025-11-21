# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))  # Importing kooplearn for autodoc

project = "kooplearn"
copyright = "2023, Pietro Novelli"
author = "Pietro Novelli"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.bibtex",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_typehints = "none"
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_book_theme"
html_logo = "../logo.svg"

html_theme_options = {
    "repository_url": "https://github.com/Machine-Learning-Dynamical-Systems/kooplearn",
    "use_repository_button": True,  # add a 'link to repository' button
    "use_issues_button": False,  # add an 'Open an Issue' button
    "path_to_docs": ("docs"),
    "show_navbar_depth": 1,
    "show_toc_level": 3,
}

html_favicon = "favicon.png"

bibtex_bibfiles = ["bibliography.bib"]

myst_enable_extensions = ["amsmath", "dollarmath", "html_image"]
nb_execution_timeout = -1
