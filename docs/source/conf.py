# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Trajectory Manifold'
copyright = '2023, Helmuth Naumer'
author = 'Helmuth Naumer'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# Add module to path
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))


# Some code stolen to fix an error in Sphinx
def no_namedtuple_attrib_docstring(app, what, name,
                                   obj, options, lines):
    is_namedtuple_docstring = (len(lines) == 1 and
        lines[0].startswith('Alias for field number'))
    if is_namedtuple_docstring:
        # We don't return, so we need to purge in-place
        del lines[:]

def setup(app):
    app.connect(
        'autodoc-process-docstring',
        no_namedtuple_attrib_docstring,
    )
