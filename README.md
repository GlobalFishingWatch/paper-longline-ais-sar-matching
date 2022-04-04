# Python Repo Template

This repo is a python project template from which you can create a new repo (see *Setup*). This template repo is meant to:

1. Make starting a new project easier. With a framework to start from, we hope this will make.
2. Help us be more consistent across projects. This allows new people (and your future self) to more efficiently orient themselves to your repo.
3. Be a gentle reminder of best practices at different stages of a project. Haven't used the `queries/` folder yet? Perhaps it's time to pull some queries out of those notebooks! Haven't put any code into your `<module>/` folder yet? Schedule some time to modularize your code.

The repo structure and a description of the intended use of each folder/file is provided in the *Structure and descriptions* section. Although you should try to stick to these conventions wherever possible, feel free to add, remove, or modify folders or add subfolder structure within folders as needed. However, when you do stray from this structure, be sure to document them in the README to that those new to your repo can still quickly understand where to find things.

This repo template is meant to be taken in the spirit of the following Pep8 guidelines:

>*“Consistency within a project is more important. Consistency within one module or function is the most important. ... However, know when to be inconsistent -- sometimes style guide recommendations just aren't applicable. When in doubt, use your best judgment. Look at other examples and decide what looks best. And don't hesitate to ask!”*

A slack channel `#prj-innovation-best-practices` is available for any and all discussion around best practices, including repo management. This channel is meant to be a place where people can ask for advice, suggest a change in best practices, share a clever solution they came up with, share a new tool they found... anything related to best practices in any language or application!

## Setup

### Creating a new repo from template

1. Create new repo with template by going to [the repo in GitHub](https://github.com/GlobalFishingWatch/research-python-template) and clicking the `Use this template` button. You can also use the GitHub command line tool, `gh repo create --template https://github.com/GlobalFishingWatch/research-python-template.git` option. More documentation on this library is available [here](https://cli.github.com/manual/gh_repo_create).
2. Rename `research_pipeline_template` to your module name. The convention is to name it the same as your repo, swapping any dashes for underscore.
3. Update the `[metadata]` section of `setup.cfg`. The `name` must match your module name in order be setup as a module (see next section).
4. Setup your python environment. We recommend using the `radenv` standard research environment as a starting point. See the [radenv repo](https://github.com/GlobalFishingWatch/radenv) for how to set this up.
4. Run `pre-commit install` to set up the pre-commit hooks for this repo. Now when you commit, they will be automatically run. You can modify what hooks to use in `.pre-commit-config.yaml`, but we suggest only modifying the arguments for a particular hook and not removing it so that we can maintain high level styling consistency across research team projects.
5. Update the `README.md` to what is relevant for your project.

### Turning your repo into a module

1. If you have not updated the `[metadata]` section of `setup.cfg`, do that now (see step 3 in previous section).
2. Add to dependencies in the `[options]` sections as needed, keeping what is already there so that styling/linting and notebooks are supported unless you are absolutely certain you don't want these.
3. Run `pip install -e .` to install the module. This will create a folder titled `<module>.egg-info` that will allow you to access the code within your `<module>` folder from outside of that folder by doing `import <module>` without any need to use paths.


## Structure and descriptions

To learn more about this structure and how it is meant to be used throughout different stages of the project, see [this presentation](https://docs.google.com/presentation/d/1E51s4VhcLzCwN_v_yeOpaGLtNEF5fcZanlO1lRsmGiw/edit?usp=sharing).

    |- README.md		<- Top-level README on how to use this repo
    |- data			    <- Data files. Create subfolders as necessary.
    |- docs			    <- Documentation.
    |- models		    <- Trained models, model summaries, etc.
    |- notebooks 	    <- Jupyter notebooks and R markdown. Naming convention TBD. Use jupytext to convert
    |				       to .py before committing and do not commit the notebook itself. Suggest having
    |				       subfolders for different subsets of analysis. Could also have different
    |				       subfolders for each person if that makes more sense for your project/team.
    |- outputs		    <- Models results, static reports, etc. Create  additional subfolders as necessary.
    |    |- figures	    <- Create versioned figures folders if desired.
    |- queries		    <- SQL files (.sql, .jinja2, etc). Create subfolders as necessary. When creating
    |				       new tables in BigQuery, be sure to use a schema with good field descriptions.
    |- scripts		    <- Regular python and R files. These may use code from <module> but are not
    |				       notebooks. A good place for scripts that run data pipelines, train models, etc.
    |- setup.cfg		<- Needed for pip install of repo.
    |- setup.py		    <- Needed for pip install of repo.
    |- tests		    <- Unit tests.
    |- <module>		    <- Source code for the project (.py). This is what is referenced after doing `pip
    |				       install -e .` Convention is that the name is the same as your repo but with
    |				       underscores instead of dashes. As the project expands, group similar code into
    |				       subfolders (submodules).
    |- .gitignore	    <- Modify as necessary.
    |- .pre-commit-config.yaml	    <- Style linting using `pre-commit library. Must run `pre-commit
    |                                  install` once to turn on commit hook. Modify as necessary.
