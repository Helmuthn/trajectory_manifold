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
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_theme_options = {
    'github_user': 'Helmuthn',
    'github_repo': 'trajectory_manifold',
    'description': 'Statistically Rigorous ODE Forecasting',
    'fixed_sidebar': True,
    'sidebar_collapse': False,
    'page_width': '80rem',
    'font_size': '1em',
}

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html'
    ]
}

html_css_files = [
    'css/custom.css',
]

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
