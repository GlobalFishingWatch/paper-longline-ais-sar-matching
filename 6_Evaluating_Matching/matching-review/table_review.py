# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
# %%
import pandas as pd

# %%
q = """SELECT
  detect_id,
  a.ssvid AS new_matched,
  b.ssvid AS brian_matched,
  score
FROM
  `world-fishing-827.proj_walmart_dark_targets.m_ksat_detections_ind_v20201124_3matched` a
JOIN
  `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201120` b
ON
  detect_id = DetectionId
WHERE
  (a.ssvid != b.ssvid
    OR (a.ssvid IS NULL
      AND b.ssvid IS NOT NULL)
    OR (b.ssvid IS NULL
      AND a.ssvid IS NOT NULL))"""

matched_comparison = pd.read_gbq(q, project_id="world-fishing-827")

# %%
q_manuals = """with
manual_matches as (
select * from
`world-fishing-827.proj_walmart_dark_targets.manual_matches_ind_v20200406`
union all
select * from
`world-fishing-827.proj_walmart_dark_targets.manual_matches_fp_v20200406`
)

select * from manual_matches"""

manual_matches = pd.read_gbq(q_manuals, project_id="world-fishing-827")

# %%
manuals_matched_inNew = matched_comparison.loc[
    matched_comparison["new_matched"].isin(
        manual_matches["manual_adj_ssvid"].astype(str)
    )
]

# %%
# Three of the previously manually matched SSVID were matched automatically in the new run.
manuals_matched_inNew

# %%
# New match that wasn't matched in previous
newMatch = matched_comparison.loc[
    ~matched_comparison["new_matched"].isna()
    & matched_comparison["brian_matched"].isna()
]
print(len(newMatch))

# %%
# previous match that wasn't matched in the new run
oldMatch = matched_comparison.loc[
    matched_comparison["new_matched"].isna()
    & ~matched_comparison["brian_matched"].isna()
]
print(len(oldMatch))

# %%
# matched in both runs, but different ssvid
diff_match = matched_comparison.loc[
    ~matched_comparison["new_matched"].isna()
    & ~matched_comparison["brian_matched"].isna()
]
print(len(diff_match))

# %%
