# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import math
import os
from datetime import datetime, timedelta

import cartopy
import cartopy.crs as ccrs
import cmocean
import geopandas as gpd
# # %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# import pyseas.rasters
import pandas as pd
# +
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

# +
# scene_id = 'RS2_20191021_015301_0074_DVWF_HH_SCS_765981_0126_30985398'

# q = f''' select * from scratch_david.walmart_rasters
# where scene_id="{scene_id}" '''

# df= pd.read_gbq(q, project_id='world-fishing-827')
# -

q = """ select
 ssvid,
 scene_id,
 a.detect_lat detect_lat,
 a.detect_lon detect_lon,
 delta_minutes,
 lat_interpolate_adjusted,
 lon_interpolate_adjusted,
 probability
 from scratch_david.walmart_rasters a
 join
 proj_walmart_dark_targets.all_detections_and_ais_v20201221 b
 using(ssvid,scene_id)
 where score > 0 and score < 1e-3 """
df = pd.read_gbq(q, project_id="world-fishing-827")

# +
# df['detect_lon'] = df['detect_lat_1']

# +
q = f"""SELECT * FROM
`world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221`
where
--scene_id = '{scene_id}'
-- and
score > 0 and score < 1e-3"""

df_matched = pd.read_gbq(q, project_id="world-fishing-827")
# -

# !mkdir low_scores

# +
dotsize = 4
min_value = 1e-9


for index, row in df_matched.iterrows():

    ssvid = row.ssvid
    scene_id = row.scene_id
    score = row.score

    print(f"score: {score:.2E}")
    print(f"vessel {ssvid}")
    print(f"scene: {scene_id}")

    d = df[(df.ssvid == ssvid) & (df.scene_id == scene_id)]
    d_matched = df_matched[
        (df_matched.ssvid == ssvid) & (df_matched.scene_id == scene_id)
    ]
    num_lats = int((d.detect_lat.max() - d.detect_lat.min()) * 200)
    num_lons = int((d.detect_lon.max() - d.detect_lon.min()) * 200)

    grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

    min_lon = d.detect_lon.min()
    min_lat = d.detect_lat.min()
    max_lon = d.detect_lon.max()
    max_lat = d.detect_lat.max()

    scale = 200

    def fill_grid(r):
        y = int((r.detect_lat - min_lat) * scale)
        x = int((r.detect_lon - min_lon) * scale)
        grid[y][x] += r.probability

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    grids = []
    lats_int = []
    lons_int = []
    minutes = d.delta_minutes.unique()

    for delta_minutes in minutes:

        d2 = d[d.delta_minutes == delta_minutes]
        grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

        def fill_grid(r):
            y = int((r.detect_lat - min_lat) * scale)
            x = int((r.detect_lon - min_lon) * scale)
            grid[y][x] += r.probability

        d2.apply(fill_grid, axis=1)

        grids.append(grid)
        lats_int.append(d2.lat_interpolate_adjusted.values[0])
        lons_int.append(d2.lon_interpolate_adjusted.values[0])

    norm = colors.LogNorm(vmin=min_value, vmax=grids[0].max())
    grids[0][grids[0] < min_value] = 0
    ax1.imshow(
        np.flipud(grids[0]), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat]
    )
    ax1.scatter(
        [lons_int[0]],
        [lats_int[0]],
        label="extrapolated position",
        s=dotsize,
        color="red",
    )
    ax1.scatter(
        d_matched.detect_lon.values,
        d_matched.detect_lat.values,
        color="black",
        label="matched detection",
        s=dotsize,
    )
    ax1.title.set_text(f"Minutes to scene: {round(minutes[0])}")

    if len(grids) > 1:
        norm = colors.LogNorm(vmin=min_value, vmax=grids[1].max())
        grids[1][grids[1] < min_value] = 0

        ax2.imshow(
            np.flipud(grids[1]), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat]
        )
        ax2.scatter(
            [lons_int[1]],
            [lats_int[1]],
            label="extrapolated position",
            s=dotsize,
            color="red",
        )
        ax2.scatter(
            d_matched.detect_lon.values,
            d_matched.detect_lat.values,
            color="black",
            label="matched detection",
            s=dotsize,
        )
        ax2.title.set_text(f"Minutes to scene: {round(minutes[1])}")

    if len(grids) > 1:
        for i in range(2):
            grids[i] = np.divide(grids[i], grids[i].sum())

        grid = np.multiply(grids[0], grids[1])
        grid = np.divide(grid, grid.sum())

    else:
        grid = np.divide(grids[0], grids[0].sum())

    grid[grid < min_value] = 0
    norm = colors.LogNorm(vmin=min_value, vmax=grid.max())
    ax3.imshow(np.flipud(grid), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat])
    ax3.scatter(
        d.lon_interpolate_adjusted.unique(),
        d.lat_interpolate_adjusted.unique(),
        label="extrapolated position",
        s=dotsize,
        color="red",
    )
    ax3.scatter(
        d_matched.detect_lon.values,
        d_matched.detect_lat.values,
        color="black",
        s=dotsize,
    )
    d = df_matched[df_matched.ssvid == ssvid]
    plt.scatter(
        d_matched.detect_lon.values,
        d_matched.detect_lat.values,
        color="black",
        label="matched detection",
        s=dotsize,
    )

    ax3.title.set_text("multiplied probability")
    plt.legend()
    fig.suptitle(f"{ssvid}, score: {score:0.2e}, scene: {scene_id}")
    plt.savefig(f"low_scores/{ssvid}_{scene_id}.png", bbox_inches="tight", dpi=300)
    plt.show()

# -

d_matched


# +
q = """with
# figure out where there is more than one seg_id in a scene
good_segs as (
--   select seg_id, scene_id from
--   (
--     select
--       scene_id, seg_id, ssvid, row_number() over (partition by seg_id, scene_id order by positions desc, min_delta_minutes) row_number
--     from
--     (
      select  seg_id, ssvid,scene_id, count(*) positions, min(abs(delta_minutes)) min_delta_minutes from (
      select distinct seg_id, ssvid, scene_id, delta_minutes from scratch_david.walmart_rasters)
      group by seg_id, ssvid, scene_id, scene_id
--     )
--   )
--   where row_number = 1
),

source_table as
(select * from scratch_david.walmart_rasters),

raster_sum1 as
(
  select
    scene_id, ssvid, detect_lat,
    detect_lon, seg_id, delta_minutes,
    probability/prob_sum as probability_norm
  from
    source_table
  join (
    select ssvid, scene_id, delta_minutes, sum(probability) prob_sum
    from source_table
    group by ssvid, scene_id, delta_minutes
  )
  using(ssvid, scene_id, delta_minutes)
),

raster_sum1_single as
(select
 scene_id, ssvid, detect_lat,
    detect_lon, seg_id,
    delta_minutes as delta_minutes_1,
    null as delta_minutes_2,
    probability_norm
from raster_sum1
join
good_segs
using(seg_id, ssvid,scene_id)
where positions  = 1
),


raster_sum1_double as
(select
 *
from raster_sum1
join
good_segs
using(seg_id, ssvid,scene_id)
where positions  = 2
),


multiplied_prob as (
  select
  a.scene_id scene_id,
  a.ssvid,
  a.seg_id,
  a.delta_minutes delta_minutes_1,
  b.delta_minutes delta_minutes_2,
  a.detect_lat,
  a.detect_lon,
  a.probability_norm*ifnull(b.probability_norm, 1) probability
  from
  raster_sum1_double a
  left join
  raster_sum1_double b
  on a.ssvid = b.ssvid and a.seg_id=b.seg_id
  and a.scene_id = b.scene_id
  and round(a.detect_lon*200) = round(b.detect_lon*200)
  and round(a.detect_lat*200) = round(b.detect_lat*200)
  where (a.delta_minutes > b.delta_minutes or b.delta_minutes is null)
),


multiplied_prob_norm as
(
  select
    scene_id, ssvid, detect_lat,
    detect_lon, seg_id,
    delta_minutes_1,
    delta_minutes_2,
    probability/prob_sum as probability_norm
  from
    multiplied_prob
  join (
    select ssvid, scene_id, delta_minutes_1, sum(probability) prob_sum
    from multiplied_prob
    group by ssvid, scene_id, delta_minutes_1
  )
  using(ssvid, scene_id, delta_minutes_1)
) ,

all_probabilities as (select * from
multiplied_prob_norm
union all
select * from raster_sum1_single),


footprints as (select distinct footprint, scene_id
 from proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 select distinct footprint, scene_id from
 proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110)

select scene_id, ssvid, sum(probability_norm) prob
from
all_probabilities
join
footprints
using(scene_id)
where st_contains( ST_GEOGFROMTEXT(footprint),st_geogpoint(detect_lon, detect_lat))
group by scene_id, ssvid


"""

df2 = pd.read_gbq(q, project_id="world-fishing-827")
# -

df.head()

# +


from random import random

xx = []
for i in range(10000):
    x = np.array(list(map(lambda x: int(float(x) > random()), df2.prob))).sum()
    xx.append(x)
# -

import seaborn as sns

sns.histplot(xx, bins=15)
plt.title("likely number of AIS vessels in scenes")

xx = np.array(xx)
xx.mean(), xx.std()
