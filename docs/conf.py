# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath('..')) # This is needed to import kooplearn

project = 'kooplearn'
copyright = '2023, Pietro Novelli, Grégoire Pacreau, Bruno Belucci, Vladimir Kostic'
author = 'Pietro Novelli, Grégoire Pacreau, Bruno Belucci, Vladimir Kostic'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',
    "myst_nb"
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

bibtex_bibfiles = ['bibliography.bib']

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "html_image"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_typehints = "none"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
#html_static_path = ['_assets']
html_theme = 'sphinx_book_theme'
html_logo = '../logo.svg'
