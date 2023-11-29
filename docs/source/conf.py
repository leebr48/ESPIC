import os
import sys
sys.path.insert(0, os.path.abspath('../../src/espic'))

project = 'ESPIC'
copyright = '2023, Patrick Kim and Brandon Lee'
author = 'Patrick Kim and Brandon Lee'

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = [".rst", ".md"]

templates_path = ['_templates']

exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = 'furo'

myst_enable_extensions = [
    "colon_fence",
]

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

html_static_path = ['_static']
