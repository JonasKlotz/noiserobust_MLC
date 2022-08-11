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
import os
import sys
from unittest.mock import MagicMock


class Mock(MagicMock):


    @classmethod
    def __getattr__(cls, name):
        return MagicMock()


class ModuleMock(Mock):
    """
    Used to add python libraries. If not added that way python modules written in C can't be imported by sphinx autodoc.

    """
    def __init__(self, path='', *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.path = path

    def __getattr__(self, name):
        if name in ('_mock_methods', '_mock_unsafe'):
            return super(ModuleMock, self).__getattr__(name)
        return ModuleMock(path=self.path + '.' + name)

    def __repr__(self):
        return self.path


# for knime_table.py, knime_schema.py
sys.path.insert(
    0, os.path.abspath(os.path.join('../..'))
)
sys.path.insert(
    0, os.path.abspath(os.path.join('../', 'src', 'lamp'))
)
sys.path.insert(
    0, os.path.abspath(os.path.join('../', 'src',))
)
print(sys.path)


# -- Project information -----------------------------------------------------

project = "RSIM PROJECT"
author = "Jonas Klotz"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser"]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# mock import that sphinx recognizes them
MOCK_MODULES = ['numpy', 'pandas', 'pyarrow']

sys.modules.update((mod_name, ModuleMock()) for mod_name in MOCK_MODULES)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = "RS.png"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'style_nav_header_background': "white",
    'collapse_navigation': False,
    'analytics_anonymize_ip': True,
}
html_js_files = [
    'js/custom.js'
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


# -- Extension configuration -------------------------------------------------
master_doc = 'index'
# html_css_files = ["custom.css"]

html_css_files = [
    'css/custom.css',
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'test', u'Test Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'Test', u'Test Documentation',
   author, 'Test', 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False

