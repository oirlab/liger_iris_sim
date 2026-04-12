# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re
import sys, os
sys.path.insert(0, os.path.abspath("../.."))

import matplotlib
matplotlib.use("Agg")

import liger_iris_sim


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'liger_iris_sim'
copyright = '2026, Liger and IRIS Data Reduction System Team'
author = 'Liger and IRIS Data Reduction System Team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
]

nbsphinx_execute = "always"

napoleon_custom_sections = [
    ('Returns', 'params_style')
]

autosummary_generate = True  # Generate autosummary stubs automatically
autodoc_default_options = {
    'members': True,          # Include all class members
    'private-members': True, # Include private members
    'undoc-members': False,    # Include undocumented members
    'show-inheritance': True, # Show class inheritance
}

templates_path = ['_templates']
exclude_patterns = []

# Add the custom CSS file
html_css_files = [
    'custom.css'
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# https://sphinxawesome.xyz/how-to/configure/
html_theme = 'sphinxawesome_theme'
#html_theme = 'sphinx_rtd_theme'
from sphinxawesome_theme.postprocess import Icons
from pygments.styles import get_all_styles
pygments_style = "friendly"
pygments_style_dark = "friendly"
html_permalinks_icon = Icons.permalinks_icon
html_static_path = ['_static']
html_css_files = ["custom.css"]

add_module_names = False

# Version
version = liger_iris_sim.__version__
# The full version, including alpha/beta/rc tags.
#release = liger_iris_sim.__version__

# URL
html_baseurl = 'https://astrobc1.github.io/liger_iris_sim/'