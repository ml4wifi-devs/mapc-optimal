# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from datetime import date

project = 'mapc_optimal'
copyright = (f'{date.today().year}, Maksymilian Wojnar,Wojciech Ciężobka,'
             f'Katarzyna Kosek-Szott,Krzysztof Rusek,Szymon Szott')
author = ('Maksymilian Wojnar,Wojciech Ciężobka,Katarzyna Kosek-Szott,'
          'Krzysztof Rusek,Szymon Szott, Piotr Chołda')
version = 'latest'

sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.autodoc.typehints',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'autoapi.extension'
              ]

templates_path = ['_templates']
exclude_patterns = []
napoleon_preprocess_types = True

github_url = "https://github.com/ml4wifi-devs/mapc-optimal"

html_static_path = ['_static']
source_suffix = ['.rst', '.md', '.ipynb']

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"
autodoc_typehints = 'none'

html_theme_options = {'show_powered_by': False,
                      'github_user': 'ml4wifi-devs',
                      'github_repo': 'mapc-optimal',
                      'github_banner': True,
                      'show_related': False,
                      'note_bg': '#FFF59C'
                      }

# AutoAPI configuration
autoapi_dirs = ["../mapc_optimal"]
autoapi_type = "python"
autoapi_add_toctree_entry = True
autoapi_options = ["show-module-summary","members", "undoc-members", 'imported-members',
                   "special-members"]
autoapi_python_class_content="both"
autodoc_class_signature = "separated"
autodoc_typehints = 'none'

# sphinx-build -b html docs docs/build/html
#  open docs/build/html/index.html
