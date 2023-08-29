# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'kooplearn'
copyright = '2023, Pietro Novelli, Grégoire Pacreau, Bruno Belucci, Vladimir Kostic'
author = 'Pietro Novelli, Grégoire Pacreau, Bruno Belucci, Vladimir Kostic'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'sphinxcontrib.bibtex',
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
    "dollarmath"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_typehints = "none"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
