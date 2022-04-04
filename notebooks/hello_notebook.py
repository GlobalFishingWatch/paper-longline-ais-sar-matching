# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

import research_python_template
from research_python_template import hello

# Use _reload() while in development so that changes
# to your module code are recognized.
research_python_template._reload()

# %%
hello.say_hello("team!")

# %%
