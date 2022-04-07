# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Imports -----------------------------------------------------------------

from asterion import __version__


# -- Project information -----------------------------------------------------

project = 'Asterion'
copyright = '2021, Alex Lyttle'
author = 'Alex Lyttle'

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc',      # Automatically generate documentation
    'sphinx.ext.napoleon',     # Support for Google-style docstrings
    # 'autodocsumm',  # Add nice summaries of classes, methods and attributes
    # 'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',  # Link to external documentation
    'sphinx.ext.viewcode',     # View source code
    'sphinx.ext.mathjax',      # Render math
    'nbsphinx',                # Generate notebooks
    # 'sphinx_inline_tabs',      # Inline tabs (introduces .. tab:: domain)
    'sphinx_search.extension',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Don't prepend module names to functions
add_module_names = False

# Suppress these warnings
suppress_warnings = ["autoapi.python_import_resolution"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "announcement": "<em>Attention</em>! This project is in early development, " + \
                    "expect frequent changes to functionality!",
    "light_logo": "images/asterion-96dpi-light.png",
    "dark_logo": "images/asterion-96dpi-dark.png",
    "sidebar_hide_name": True,
    "light_css_variables": {
        # Legible font for better accessibility
        "font-stack": "Atkinson Hyperlegible, sans-serif",
        "font-stack--monospace": "Iosevka Hyperlegible Web, monospace",
    },
}

html_title = f"{project} v{release}"
html_short_title = project

html_css_files = [
    'css/custom.css',
]


# -- Autodoc options ---------------------------------------------------------

autodoc_typehints = 'none'  # show type hints in doc body when not 
                                   # specified in docstrings
# autodoc_typehints_description_target = 'documented'
autodoc_inherit_docstrings = True
# autodoc_default_options = {
    # -- autodocsumm options -------------------------------------------------
    # 'autosummary': True,  # Add summaries automatically
    # 'autosummary-nosignatures': True,
    # 'autosummary-sections': 'Classes ;; Functions ;; Data',
# }
# autodoc_type_aliases = {
    # "DistLike": "asterion.annotations.DistLike",
#     "ArrayLike": "numpy.typing.ArrayLike",
# }
# autosummary_generate = True
autodoc_typehints_format = 'short'


# -- InterSphinx options -----------------------------------------------------

intersphinx_mapping = {
    "arviz": ("https://arviz-devs.github.io/arviz/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "corner": ("https://corner.readthedocs.io/en/stable/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/dev/", None),
}


# Napoleon options -----------------------------------------------------------

# napoleon_type_aliases = {}
# napoleon_preprocess_types = True
# napoleon_attr_annotations = True
napoleon_numpy_docstring = False


# -- NBSphinx options --------------------------------------------------------

nbsphinx_execute = "never"


# -- AutoAPI options ---------------------------------------------------------

autoapi_dirs = ['../../asterion']
autoapi_root = 'guide/api'
autoapi_options = ['members', 'show-inheritance', 'imported-members']
autoapi_member_order = 'groupwise'
