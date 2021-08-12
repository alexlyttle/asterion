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

import asterion
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
    'sphinx.ext.napoleon',  # Add napoleon to the extensions list
    'sphinx.ext.autodoc',
    # 'sphinx.ext.linkcode',  # Uncomment for links to source code on GitHub
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "announcement": "<em>Attention</em>! This project is an alpha build, " + \
                    "please use with caution.",
}

html_title = f"{project} v{release}"
html_short_title = project


# -- Autodoc options ---------------------------------------------------------

autodoc_typehints = 'description'  # show type hints in doc body
autodoc_typehints_description_target = 'documented'

# -- Link code options -------------------------------------------------------

# Uncomment the following for links to source in GitHub
# def linkcode_resolve(domain, info):
#     def find_source():
#         # try to find the file and line number, based on code from numpy:
#         # https://github.com/numpy/numpy/blob/master/doc/source/conf.py#L286
#         obj = sys.modules[info['module']]
#         for part in info['fullname'].split('.'):
#             obj = getattr(obj, part)
#         import inspect
#         import os
#         fn = inspect.getsourcefile(obj)
#         fn = os.path.relpath(
#             fn, 
#             tart=os.path.dirname(asterion.__file__)
#         )
#         source, lineno = inspect.getsourcelines(obj)
#         return fn, lineno, lineno + len(source) - 1

#     if domain != 'py' or not info['module']:
#         return None
#     try:
#         filename = 'asterion/%s#L%d-L%d' % find_source()
#     except Exception:
#         return None
#     tag = 'main' if 'dev' in release else ('v' + release)
#     return "https://github.com/alexlyttle/helium-glitch-fitter/blob/%s/%s" \
#         % (tag, filename)


# -- InterSphinx options -----------------------------------------------------

intersphinx_mapping = {
    "arviz": ("https://arviz-devs.github.io/arviz", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
}


# Napoleon options -----------------------------------------------------------

napoleon_type_aliases = {
    "Array1D": "asterion.annotations.Array1D",
    "Array2D": "asterion.annotations.Array2D",
    "Array3D": "asterion.annotations.Array3D",
}
