templates_path = ["_templates"]
source_suffix = [".rst", ".md"]

master_doc = "index"

project = "TileDB-CF-Py"
copyright = "2021, TileDB, Inc"
author = "TileDB, Inc"
release = "0.6.0"
version = "0.6.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
]


language = None
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"
todo_include_todos = False


# -- Options for HTML output -------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for HTMLHelp output ---------------------------------------
htmlhelp_basename = "tiledb-cf-doc"

# -- Options for LaTeX output ------------------------------------------
latex_documents = [
    (
        master_doc,
        "tiledb.cf.tex",
        "TileDB-CF-Py Documentation",
        "TileDB, Inc",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------
man_pages = [(master_doc, "tiledb.cf", "TileDB-CF-Py Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------
texinfo_documents = [
    (
        master_doc,
        "tiledb.cf",
        "TileDB-CF-Py Documentation",
        author,
        "tiledb.cf",
        "One line description of project.",
        "Miscellaneous",
    ),
]
