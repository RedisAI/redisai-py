project = "redisai-py"
copyright = "2020, RedisLabs"
author = "RedisLabs"
release = "1.0.1"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = "%s-%s" % (project, release)

napoleon_use_ivar = True
napoleon_use_rtype = True
napoleon_use_param = True
napoleon_include_init_with_doc = True

add_module_names = False
doctest_test_doctest_blocks = None
autoclass_content = "class"
