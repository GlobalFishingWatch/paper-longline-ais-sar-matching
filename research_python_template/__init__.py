def _reload():
    """
    Reload modules during development. Note that this
    will not reload any changes made to this function
    or anywhere else in this __init__.py file.

    Add modules to reload as you create them.
    When you add modules or make any changes in this file, you need
    to run these two lines of code in any script where you are using
    this library. They must run be after you import your library but
    before you do `from ... import ...` statements.
        >>> import importlib
        >>> importlib.reload(research_python_template)
    """
    import importlib

    import research_python_template
    from research_python_template import config, hello

    importlib.reload(research_python_template)
    importlib.reload(hello)
    importlib.reload(config)
