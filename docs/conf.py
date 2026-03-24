"""Sphinx configuration for dynaris documentation."""

import os
import sys

# Add source to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "dynaris"
copyright = "2025, quantsci"
author = "quantsci"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_theme_options = {
    "description": "JAX-powered Dynamic Linear Models",
    "github_user": "quant-sci",
    "github_repo": "dynaris",
    "github_button": True,
    "fixed_sidebar": True,
    "sidebar_collapse": True,
    "page_width": "940px",
    "sidebar_width": "220px",
}
html_static_path = ["_static"]

# -- Extension configuration -------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Mock imports for autodoc (JAX may not be available in docs build)
autodoc_mock_imports = ["jax", "jaxlib"]
