# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# # Match SAR Detections

# %%
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml


def gbq(q):
    return pd.read_gbq(q)


# this is a variable that can be changed to make different experiments
# to test how matching does when the positions are not close in time
# to the scene
min_hours_to_scene = 0


# %%

    
    
def replace_period_colon(table_id):
    t = table_id.split(".")
    if len(t)==3:
        return f"{t[0]}:{t[1]}.{t[2]}"
    else:
        return table_id


# %% [markdown]
# ## Key Variables
#

# %%
with open("ksat_matching_v20210426.yaml", "r") as f:
    y = yaml.load(f, Loader=yaml.FullLoader)


# %%
y

# %%
detections_table = y["detections_table"]
norad_id = y["norad_id"]
use_vms = y["use_vms"]
output_dataset = y["output_dataset"]
version_name = y["version_name"]
research_pipe = y["research_pipe"]
segs_table = y["segs_table"]
vessel_info_table = y["vessel_info_table"]

if use_vms:
    vms_datasets = y["vms_tables"]
    matches_scored_vms_tables = []
    for v in vms_datasets:
        v = v.split("_")[1]
        matches_scored_vms_tables.append(
            "{}.m_{}_1scored_vms_{}".format(output_dataset, version_name, v)
        )

extrapolated_table = "{}.{}_1_extrapolated_ais".format(output_dataset, version_name)
scored_ais_table = "{}.{}_2_scored".format(output_dataset, version_name)
ranked_table = "{}.{}_3_ranked".format(output_dataset, version_name)
matched_table = "{}.{}_4_matched".format(output_dataset, version_name)


# %%
scored_ais_table

# %% [markdown]
# ## Get the dates of the detections

# %%
q = f"""select
distinct date(detect_timestamp) date
from (select TIMESTAMP_ADD(ProductStartTime, INTERVAL
cast(timestamp_diff(ProductStopTime, ProductStopTime,
SECOND)/2 as int64) SECOND) detect_timestamp
from ({detections_table})) """
print(q)
df = pd.read_gbq(q)
df.head()

# %%
the_dates = []
for d in df.date.values:
    d = pd.to_datetime(d)
    the_dates.append((d - timedelta(days=1)).strftime("%Y-%m-%d"))
    the_dates.append(d.strftime("%Y-%m-%d"))
    the_dates.append((d + timedelta(days=1)).strftime("%Y-%m-%d"))
the_dates = list(set(the_dates))


# %%
start_date = pd.to_datetime(df.date.min()).strftime("%Y-%m-%d")
end_date = pd.to_datetime(df.date.max()).strftime("%Y-%m-%d")
start_date, end_date


# %% [markdown]
# ## Update Satellite Table if Need Be
#
# Because the location of vessel detections can be offset due to the doppler shift, we have to know the position and velocity of the SAR satellites

# %%
for the_date in the_dates:
    """this checks to see if there are satellite positions
    for this norad_id on a given date... and if there isn't
    it will run a script to update the satellite positions"""

    sat_positions_table = (
        "satellite_positions_v20190208.sat_noradid_{norad_id}_{YYYYMMDD}".format(
            norad_id=norad_id, YYYYMMDD=the_date.replace("-", "")
        )
    )
    print(sat_positions_table)
    dataset = sat_positions_table.split(".")[0]
    table = sat_positions_table.split(".")[1]
    q = """SELECT size_bytes FROM {dataset}.__TABLES__ WHERE table_id='{table}' """.format(
        dataset=dataset, table=table
    )
    df = pd.read_gbq(q, project_id="world-fishing-827")
    if (len(df)) == 0:
        command = "python sat_positions.py {} {}".format(norad_id, the_date)
        print(command)
        os.system(command)

# %%


# %%
# end_date = "2019-08-12"

# %% [markdown]
# ## Extrapolate and Match AIS

# %%

command = f"""jinja2  01_extrapolate_ais.sql.j2 \
   -D start_date="{start_date}" \
   -D end_date="{end_date}" \
   -D norad_id="{norad_id}" \
   -D segs_table="{segs_table}" \
   -D research_pipe="{research_pipe}" \
   -D vessel_info_table="{vessel_info_table}" \
   -D min_hours_to_scene="{min_hours_to_scene}" \
   -D detections_table="{detections_table}"  | \
   bq query --replace \
   --destination_table={replace_period_colon(extrapolated_table)}\
   --allow_large_results --use_legacy_sql=false """

# %%
os.system(command)

# %% [markdown]
# ## Score Detections

# %%
# this is here because of a versin of this code that includes vms
matches_scored_vms_tables_unioned = ""


command = f"""jinja2  02_score_detections.sql.j2 \
    -D extrapolated_table="{extrapolated_table}" \
    -D detections_table="{detections_table}"  \
    -D vessel_info_table="{vessel_info_table}" \
    -D matches_scored_vms_tables="{matches_scored_vms_tables_unioned}"| \
    bq query --replace \
    --destination_table={replace_period_colon(scored_ais_table)} \
    --allow_large_results --use_legacy_sql=false """

# %%
print(command)

# %%
os.system(command)

# %%

# %% [markdown]
# ## Rank Detections

# %%
command = f"""jinja2  03_rank_detections.sql.j2 \
    -D scored_table="{scored_ais_table}"| \
    bq query --replace \
    --destination_table={replace_period_colon(ranked_table)}\
    --allow_large_results \
    --use_legacy_sql=false """

# %%
print(command)

# %%
os.system(command)

# %% [markdown]
# ## Pick Best Matches

# %%

# %%
command = f"""jinja2  04_match_detections.sql.j2 \
-D ranked_table="{ranked_table}" \
-D detections_table="{detections_table}"| \
bq query --replace \
--destination_table={replace_period_colon(matched_table)} \
--allow_large_results \
--use_legacy_sql=false """

# %%
print(command)

# %%
os.system(command)

# %% [markdown]
# # Now Calculate Multiplied Scores

# %%
scored_ais_table
scored_multiplied_ais_table = scored_ais_table + "_mult"

# %%
command = f"""jinja2  02_score_detections_multiplied.sql.j2 \
    -D extrapolated_table="{extrapolated_table}" | \
    bq query --replace \
    --destination_table={replace_period_colon(scored_multiplied_ais_table)} \
    --allow_large_results --use_legacy_sql=false """

# %%
print(command)

# %%
# # !jinja2  02_score_detections_multiplied.sql.j2     -D extrapolated_table="proj_walmart_dark_targets.matching_v20210421_1_extrapolated_ais"

# %%
os.system(command)

# %% [markdown]
# ### Rank
#

# %%
ranked_table_mult = ranked_table + "_mult"

# %%
command = f"""jinja2  03_rank_detections.sql.j2 \
    -D scored_table="{scored_multiplied_ais_table}"| \
    bq query --replace \
    --destination_table={replace_period_colon(ranked_table_mult)}\
    --allow_large_results \
    --use_legacy_sql=false """

# %%
print(command)

# %%
# # !jinja2  03_rank_detections.sql.j2     -D scored_table="scratch_david.test_ksat_detections_v20210421_2_scored_mult"

os.system(command)

# %% [markdown]
# ### Top matches

# %%
matched_table_mult = matched_table + "_mult"

command = f"""jinja2  04_match_detections.sql.j2 \
-D ranked_table="{ranked_table_mult}" \
-D detections_table="{detections_table}"| \
bq query --replace \
--destination_table={replace_period_colon(matched_table_mult)} \
--allow_large_results \
--use_legacy_sql=false """

# %%
print(command)

# %%
os.system(command)

# %% [markdown]
# # Now Use Naive Distance Metric from Most Likely Location
#
# ### Score

# %%
scored_dist_ais_table = scored_ais_table + "_dist"

# %%
command = f"""jinja2  02_score_detections_dist.sql.j2 \
    -D extrapolated_table="{extrapolated_table}" \
    -D detections_table="{detections_table}"| \
    bq query --replace \
    --destination_table={replace_period_colon(scored_dist_ais_table)} \
    --allow_large_results --use_legacy_sql=false """

# %%
print(command)

# %%
# # !jinja2  02_score_detections_dist.sql.j2     -D extrapolated_table="scratch_david.test_ksat_detections_v20210421_1_extrapolated_ais"     -D detections_table="(select ProductStartTime,ProductStopTime,scene_id,footprint,lon,lat,DetectionID from proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117 union all select ProductStartTime,ProductStopTime,scene_id,footprint,lon,lat,DetectionID from proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110)"

# %%
os.system(command)

# %% [markdown]
# ### Rank

# %%

# %%
ranked_table_dist = ranked_table + "_dist"

command = f"""jinja2  03_rank_detections.sql.j2 \
    -D scored_table="{scored_dist_ais_table}"| \
    bq query --replace \
    --destination_table={replace_period_colon(ranked_table_dist)}\
    --allow_large_results \
    --use_legacy_sql=false """

# %%
print(command)

# %%
# # !jinja2  03_rank_detections.sql.j2     -D scored_table="scratch_david.test_ksat_detections_v20210421_2_scored_dist"

# %%
os.system(command)

# %% [markdown]
# ### Top matches

# %%
matched_table_dist = matched_table + "_dist"

command = f"""jinja2  04_match_detections.sql.j2 \
-D ranked_table="{ranked_table_dist}" \
-D detections_table="{detections_table}"| \
bq query --replace \
--destination_table={replace_period_colon(matched_table_dist)} \
--allow_large_results \
--use_legacy_sql=false """

# %%
os.system(command)

# %%
matched_table_dist

# %%

# %% [markdown]
# # Create a Hybrid of Averaged and Multiplied

# %%
# extrapolated_table

# %%
# scored_multiplied_ais_table

# %% [markdown]
# ### Create Hybrid Score

# %%

scored_table_combined = scored_ais_table + "_combined"

command = f"""jinja2  02_score_combine_mult_ave.sql.j2 \
-D extrapolated_table="{extrapolated_table}" \
-D scored_ais_table_mult="{scored_multiplied_ais_table}" \
-D scored_ais_table="{scored_ais_table}"| \
bq query --replace \
--destination_table={replace_period_colon(scored_table_combined)} \
--allow_large_results \
--use_legacy_sql=false """

# %%
os.system(command)

# %% [markdown]
# ### rank

# %%
ranked_table_combined = ranked_table + "_combined"

command = f"""jinja2  03_rank_detections.sql.j2 \
    -D scored_table="{scored_table_combined}"| \
    bq query --replace \
    --destination_table={replace_period_colon(ranked_table_combined)}\
    --allow_large_results \
    --use_legacy_sql=false """

# %%
os.system(command)

# %% [markdown]
# ### Top Matches

# %%
matched_table_combined = matched_table + "_combined"

command = f"""jinja2  04_match_detections.sql.j2 \
-D ranked_table="{ranked_table_combined}" \
-D detections_table="{detections_table}"| \
bq query --replace \
--destination_table={replace_period_colon(matched_table_combined)} \
--allow_large_results \
--use_legacy_sql=false """

# %%
os.system(command)

# %%
