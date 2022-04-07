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
# # Does SAR detect gear?
#
# Feb 20 2021
# Does SAR detect gear? We have over 1,000 instances of gear in the scenes we imaged... and some of them matched to detections in radar. But they could be dark vessels near broadcasting gear (say, the vessel picking up or deploying the gear, which may not be broadcasting), or the gear might be on the deck of a boat. If the detection rate is at all significant, it will throw off our analysis.
#
#
# Key finding:
# It looks like gear almost never (<2%) matches to a sar detection if the gear is moving at under 1 knot. But the match rate increases if it is moving faster. Gear doesn't have a motor; when it is moving quickly it has to be on a vessel. Thus, we conclude that gear is almost never detected by radar and we can ignore it.
#
# import math
# import os
# from datetime import datetime, timedelta
#
# import cartopy
# import cartopy.crs as ccrs
# import cmocean
# import geopandas as gpd
# %matplotlib inline
# import matplotlib.pyplot as plt
# import numpy as np
# import pyseas.rasters
# import pandas as pd
# %%
import pyseas
# import pyseas.colors
import pyseas.cm
import pyseas.maps
import pyseas.maps.rasters
import pyseas.styles
import shapely
from cartopy import config
from matplotlib import colorbar, colors
from shapely import wkt
from shapely.geometry import Point

import ais_sar_matching.sar_analysis as sarm

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# ## Update Gear Review
#
# 228368700 - is a french patrol boat
# 412549103 - longline
# 416005715 - longline
# 4168888 - longline from Taiwan, most likely
# 100900000 - it is multiple vessels. During our time period it looks like a drifting longline

# %%

# %%
q = """with gear_in_scene as

(select * from `world-fishing-827.scratch_david.interp_test`
join
(select distinct ssvid from
`global-fishing-watch.paper_longline_ais_sar_matching.all_detections_and_ais_v20201221` where  gear
and cast(ssvid as int64) not in (
228368700 ,
412549103, 416005715 ,
4168888 ,
100900000  )

)
using(ssvid)
where within_footprint_5km
and (delta_minutes1 < 1 or delta_minutes2 > -1)
),

max_scores as (
select ssvid, scene_id, max(score) max_score from
( select ssvid, scene_id, if(not is_single and a_probability = 0 or b_probability = 0, 0, score ) score from
`world-fishing-827.scratch_david.score_test`) group by scene_id, ssvid
)

select ssvid, scene_id, ifnull(max_score,0) max_scores
 from gear_in_scene
left join max_scores
using(ssvid, scene_id)"""

df = sarm.gbq(q)

# %%
df.head()

# %%
len(df)

# %%
len(df[df.max_scores < 1e-5]) / len(df)

# %%
df[df.max_scores < 1e-3]

# %%
q = """with gear_in_scene as

(select * from `world-fishing-827.scratch_david.interp_test`
join
(select distinct ssvid from
`world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221` where gear
and cast(ssvid as int64) not in (
228368700 ,
412549103, 416005715 ,
4168888 ,
100900000  )
)
using(ssvid)
where within_footprint_5km
),

max_scores as (
select ssvid, scene_id, max(score) max_score from
( select ssvid, scene_id, if(not is_single and a_probability = 0 or b_probability = 0, 0, score ) score from
`world-fishing-827.scratch_david.score_test`) group by scene_id, ssvid
)

select ssvid, scene_id, ifnull(max_score,0) max_scores, speed1, speed2, delta_minutes1, delta_minutes2
 from gear_in_scene
left join max_scores
using(ssvid, scene_id)
order by max_scores desc"""

df = sarm.gbq(q)

# %%
df = df.dropna()
df.head(20)

# %%


df["dm"] = df.apply(lambda x: min(x.delta_minutes1, abs(x.delta_minutes2)), axis=1)
df["speed"] = df.apply(
    lambda x: x.speed1 if x.delta_minutes1 < abs(x.delta_minutes2) else x.speed2, axis=1
)

# %%
df.head()

# %%
len(df)

# %%
d = df[df.dm < 5]
d.head()

# %%
d = df[df.dm < 5]
len(d[d.max_scores > 1e-5]) / len(d), len(d)

# %%
d = df[(df.dm < 5) & (df.speed < 1)]
len(d[d.max_scores > 1e-5]) / len(d), len(d)

# %%
d = df[(df.dm < 5) & (df.speed < 0.5)]
len(d[d.max_scores > 1e-5]) / len(d), len(d)

# %%
d = df[(df.dm < 5) & (df.speed > 5)]
len(d[d.max_scores > 1e-5]) / len(d), len(d)

# %%

# %%
d = df[df.dm < 10]
len(d)

# %%
d = df[df.dm < 10]
len(d)
d["speed_floor"] = d.speed.apply(lambda x: int(x))
d["detected"] = d.max_scores.apply(lambda x: x > 1e-5)

# %%
d = d.groupby("speed_floor").mean()
d.head(10)

# %%
plt.scatter(d.index + 0.5, d.detected)
plt.xlim(0, 10)
plt.xlabel("speed of gear")
plt.ylabel("fraction that were detected by radar")
plt.title("When gear is detected, it is moving\nand thus likely on the deck of a boat")

# %%
d = df[df.dm < 10]
len(d)
d["speed_floor"] = d.speed.apply(lambda x: int(x))
d["detected"] = d.max_scores.apply(lambda x: x > 1e-5)
d = d.groupby("speed_floor").count()
d.head(10)

# %%
q = """with gear_in_scene as

(select * from `world-fishing-827.scratch_david.interp_test`
join
(select distinct ssvid from
`world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221` where not gear)
using(ssvid)
where within_footprint_5km
),

max_scores as (
select ssvid, scene_id, max(score) max_score from
( select ssvid, scene_id, if(not is_single and a_probability = 0 or b_probability = 0, 0, score ) score from
`world-fishing-827.scratch_david.score_test`) group by scene_id, ssvid
)

select ssvid, scene_id, ifnull(max_score,0) max_scores, speed1, speed2, delta_minutes1, delta_minutes2
 from gear_in_scene
left join max_scores
using(ssvid, scene_id)
order by max_scores desc"""

df = sarm.gbq(q)

# %%

df = df.dropna()
df["dm"] = df.apply(lambda x: min(x.delta_minutes1, abs(x.delta_minutes2)), axis=1)
df["speed"] = df.apply(
    lambda x: x.speed1 if x.delta_minutes1 < abs(x.delta_minutes2) else x.speed2, axis=1
)
d = df[df.dm < 10]
len(d)
d["speed_floor"] = d.speed.apply(lambda x: int(x))
d["detected"] = d.max_scores.apply(lambda x: x > 1e-5)
d = d.groupby("speed_floor").mean()
d.head(10)

# %%
plt.scatter(d.index + 0.5, d.detected)
plt.xlim(0, 15)
plt.xlabel("speed of vessel")
plt.ylabel("fraction that were detected by radar")
plt.title(
    "things that are vessels match at the same rate regardless of speed\n\
(although something weird is happening at 6 knots))"
)

# %%
d = df[df.dm < 10]
len(d)
d["speed_floor"] = d.speed.apply(lambda x: int(x))
d["detected"] = d.max_scores.apply(lambda x: x > 1e-5)
d = d.groupby("speed_floor").count()

plt.scatter(d.index + 0.5, d.detected)
plt.xlim(0, 15)
plt.xlabel("speed of vessel")
plt.ylabel("number")
plt.title("vessels by speed")

# %% [markdown]
#
