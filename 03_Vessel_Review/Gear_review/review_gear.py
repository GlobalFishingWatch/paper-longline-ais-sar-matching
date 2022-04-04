# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Review Gear
#
# Feb 20 2021
# The goal of this notebook is to review all the mmsi that Brian Wong had labeled as "gear" and make sure that they are, in fact, gear.
#
#
# Key Finding:
#
# The following mmsi, after reviewing their tracks (below), and definitely not gear:
#  - 100900000 (multiple vessels using the same mmsi, have to review it for just our area of interest)
#  - 228368700
#  - 412549103
#  - 416005715
#  - 4168888
#
#

import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyseas
import pyseas.cm
import pyseas.maps
import pyseas.styles
from matplotlib import colorbar, colors
from pyseas import maps, styles
# +
from pyseas.contrib import plot_tracks

# %matplotlib inline


def gbq(q):
    return pd.read_gbq(q, project_id="world-fishing-827")


# -


def plot_track_speed(df):
    """
    Function to plot vessels track points colored by speed, with a histogram of speed
    """

    with pyseas.context(styles.light):
        for i in df.ssvid.unique():

            d = df[df["ssvid"] == i]

            fig = plt.figure(
                figsize=(10, 14),
            )
            gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
            projinfo = plot_tracks.find_projection(df.lon, df.lat)

            ax = maps.create_map(
                gs[0]
            )  # projection=projinfo.projection, extent=projinfo.extent)
            ax.set_global()
            maps.add_land()
            maps.add_countries()
            maps.add_eezs()
            maps.add_gridlines()
            maps.add_gridlabels()

            cm = plt.cm.get_cmap("RdYlBu_r")
            z = np.array(d.speed_knots)

            normalize = mcol.Normalize(vmin=0, vmax=10, clip=True)
            zero = ax.scatter(
                d.lon.values,
                d.lat.values,
                c=z,
                cmap=cm,
                norm=normalize,
                alpha=0.7,
                transform=maps.identity,
            )

            cbar0 = plt.colorbar(zero)
            cbar0.set_label("Speed", rotation=270)

            ax1 = maps.create_map(
                gs[1], projection=projinfo.projection, extent=projinfo.extent
            )
            maps.add_land()
            maps.add_countries()
            maps.add_eezs()
            maps.add_gridlines()
            maps.add_gridlabels()

            one = ax1.scatter(
                d.lon.values,
                d.lat.values,
                c=z,
                cmap=cm,
                norm=normalize,
                alpha=0.7,
                transform=maps.identity,
            )

            cbar1 = plt.colorbar(one)
            cbar1.set_label("Speed", rotation=270)

            ax2 = fig.add_subplot(gs[2])

            Q1 = np.quantile(d.speed_knots, 0.25)
            Q3 = np.quantile(d.speed_knots, 0.75)
            IQR = Q3 - Q1
            df_noOutliers = d.speed_knots[
                ~(
                    (d.speed_knots < (Q1 - 1.5 * IQR))
                    | (d.speed_knots > (Q3 + 1.5 * IQR))
                )
            ]

            ax2.hist(d.speed_knots, bins="auto")
            plt.xlabel("Speed")
            plt.ylabel("Count")

            print(i)
            plt.show()
            print("\n")


# +
q = """with gear_in_scene as

(select * from `world-fishing-827.scratch_david.interp_test`
join
(select distinct ssvid from `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221` where gear)
using(ssvid)
where within_footprint_5km),

max_scores as (
select ssvid, scene_id, max(score) max_score from
( select ssvid, scene_id, if(not is_single and a_probability = 0 or b_probability = 0, 0, score ) score from
`world-fishing-827.scratch_david.score_test`) group by scene_id, ssvid
)

select ssvid, scene_id, ifnull(max_score,0) max_score
 from gear_in_scene
left join max_scores
using(ssvid, scene_id)"""

df = gbq(q)
# -

df.head()

df.ssvid.nunique()

len(df[df.max_score < 1e-5]) / len(df)

len(df[df.max_score < 1e-4]) / len(df)

len(df[df.max_score < 1e-3]) / len(df)

# +
q = """with gear_in_scene as

(select * from `world-fishing-827.scratch_david.interp_test`
join
(select distinct ssvid from `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221` where gear)
using(ssvid)
where within_footprint_5km),


top_matches as (
select scene_id, ssvid, score from `world-fishing-827.scratch_david.top_test`)

select scene_id, lon1, ifnull(score,0) score from gear_in_scene
left join
top_matches
using(ssvid, scene_id)

# max_scores as (
# select ssvid, scene_id, max(score) max_score from
# ( select ssvid, scene_id, if(not is_single and a_probability = 0 or b_probability = 0, 0, score ) score from
# `world-fishing-827.scratch_david.score_test`) group by scene_id, ssvid
# )

# select ssvid, scene_id, ifnull(max_score,0) max_score
#  from gear_in_scene
# left join max_scores
# using(ssvid, scene_id)"""

df = gbq(q)
# -

df.head()

len(df[df.score > 1e-5]) / len(df)

len(df[df.score > 1e-3]) / len(df)

d = df[df.lon1 < 0]
len(d[df.score > 1e-5]) / len(d)

d = df[df.lon1 > 0]
len(d[df.score > 1e-5]) / len(d)

# # Is there any gear that is seen frequently?

# +
q = """with gear_in_scene as

(select * from `world-fishing-827.scratch_david.interp_test`
join
(select distinct ssvid from `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221`
where gear)
using(ssvid)
where within_footprint_5km),


top_matches as (
select scene_id, ssvid, score from `world-fishing-827.scratch_david.top_test`)


select ssvid, avg(score) avg_score, sum(if(score>1e-5,1,0))/count(*) frac_seen, count(*) times_in_scene
from
(select ssvid, scene_id, lon1, ifnull(score,0) score from gear_in_scene
left join
top_matches
using(ssvid, scene_id))
group by ssvid order by times_in_scene desc

"""

df2 = gbq(q)
# -

df2.head()

df2[(df2.frac_seen > 0.5) & (df2.times_in_scene > 2)]

# ## Are these real vessels???
#
# 98345876 has name "net mark" and is inferred to be gear strongly
#
# 994161393 has voltage changing frequently, and is likely gear
#
# 1004 is not inferred to be gear and doesn't have name of gear... it is inferred to be a squid jigger. Let's map it's activity...

df2[(df2.frac_seen > 0.3) & (df2.times_in_scene > 2)]


# # let's map the activity of these vessels...

# +
# expensive query!
q = """
select ssvid, seg_id, timestamp, lat, lon, speed_knots
from `gfw_research.pipe_v20201001`
where seg_id in (select seg_id from `gfw_research.pipe_v20201001_segs`
where good_seg)
and ssvid in (select distinct ssvid
               from `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221`
            where gear
            union all
            select '50416013')
and date(_partitiontime) between "2019-08-09" and "2020-01-31"
order by timestamp
"""

df_gear = gbq(q)
# -
df_gear.head()

plot_track_speed(df_gear)
