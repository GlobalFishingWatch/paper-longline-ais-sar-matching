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
# # Review ambigous matches - AIS that could have matched to multiple SAR detections, or SAR to multiple vessels broadcasting AIS.
#
# from datetime import datetime, timedelta
#
# import matplotlib.cm as cm
# import matplotlib.colors as mcol
# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pyseas
# import pyseas.cm
# import pyseas.maps
# import pyseas.styles
# from matplotlib import colorbar, colors
# from pyseas import maps, styles
# %%
from pyseas.contrib import plot_tracks
from datetime import date
import matplotlib.pyplot as plt
import ais_sar_matching.sar_analysis as sarm

# %matplotlib inline


# %load_ext autoreload
# %autoreload 2

# %%
q = """with

AOI as (
SELECT distinct 'indian' as region,  footprint as footprint from  `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110`
union all
select distinct 'pacific' as region,  footprint as footprint from `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117`
),

footprints as (
select region, ST_UNION_AGG(ST_GEOGFROMTEXT(footprint)) footprint
from AOI
group by region)

select ssvid, timestamp, lat, lon, speed_knots, 2019 as year
from `world-fishing-827.gfw_research.pipe_v20201001`
where
_partitiontime between '2019-02-01' and '2020-01-20'
and ssvid in ('664089000')
--and (ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'pacific'))
--or ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'indian')))
order by timestamp


"""
df = sarm.gbq(q)

# %%
df

# %%
sarm.plot_track_speed_year(df)

# %%
q = """with

AOI as (
SELECT distinct 'indian' as region,
footprint as footprint
from  `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110`
union all
select distinct 'pacific' as region,
footprint as footprint
from `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117`
),

footprints as (
select region, ST_UNION_AGG(ST_GEOGFROMTEXT(footprint)) footprint
from AOI
group by region)

select ssvid, timestamp, lat, lon, speed_knots, 2019 as year
from `world-fishing-827.gfw_research.pipe_v20201001`
where
_partitiontime between '2019-08-01' and '2020-01-20'
and ssvid in ('664089000', '416004879')
and (ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'pacific'))
or ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'indian')))
order by timestamp


"""
df = sarm.gbq(q)

# %%
sarm.plot_double_track_speed(df)

# %%
mind = date(2019, 8, 15)
maxd = date(2020, 1, 15)

miny = 54
maxy = 59

minx = -14
maxx = -11

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lon, label=ssvid, s=1)
plt.legend(frameon=False)
plt.title("Longitude of Two MMSI")
plt.xlim(mind, maxd)
plt.ylim(miny, maxy)
plt.show()

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lat, label=ssvid, s=1)
plt.legend()
plt.title("Latitude of Two MMSI")
plt.xlim(mind, maxd)
plt.ylim(minx, maxx)

plt.show()

# %%

# %%
q = """with

AOI as (
SELECT distinct 'indian' as region,  footprint as footprint from  `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110`
union all
select distinct 'pacific' as region,  footprint as footprint from `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117`
),

footprints as (
select region, ST_UNION_AGG(ST_GEOGFROMTEXT(footprint)) footprint
from AOI
group by region)

select ssvid, timestamp, lat, lon, speed_knots, 2019 as year
from `world-fishing-827.gfw_research.pipe_v20201001`
where
_partitiontime between '2019-08-01' and '2020-01-20'
and ssvid in ('412482958', '412482984')
 --and (ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'pacific'))
--or ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'indian')))
and lat < -5
order by timestamp


"""
df = sarm.gbq(q)

# %%
mind = date(2019, 8, 15)
maxd = date(2020, 1, 15)

miny = 54
maxy = 65

minx = -14
maxx = -11

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lon, label=ssvid, s=1)
plt.legend(frameon=False)
plt.title("Longitude of Two MMSI")
plt.xlim(mind, maxd)
plt.ylim(miny, maxy)
plt.show()

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lat, label=ssvid, s=1)
plt.legend()
plt.title("Latitude of Two MMSI")
plt.xlim(mind, maxd)
# plt.ylim(minx,maxx)

plt.show()

# %%
mind = date(2019, 11, 15)
maxd = date(2019, 12, 1)

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lon, label=ssvid, s=1)
plt.legend(frameon=False)
plt.title("Longitude of Two MMSI")
plt.xlim(mind, maxd)
plt.ylim(miny, maxy)
plt.show()

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lat, label=ssvid, s=1)
plt.legend()
plt.title("Latitude of Two MMSI")
plt.xlim(mind, maxd)

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.speed_knots, label=ssvid, s=1)
plt.legend(frameon=False)
plt.title("Speed of Two MMSI")
plt.xlim(mind, maxd)


# %%


q = """with

AOI as (
SELECT distinct 'indian' as region,  footprint as footprint from  `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110`
union all
select distinct 'pacific' as region,  footprint as footprint from `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117`
),

footprints as (
select region, ST_UNION_AGG(ST_GEOGFROMTEXT(footprint)) footprint
from AOI
group by region)

select ssvid, timestamp, lat, lon, speed_knots, 2019 as year
from `world-fishing-827.gfw_research.pipe_v20201001`
where
_partitiontime between '2019-08-01' and '2020-01-20'
and ssvid in ('412329516', '412329514')
 --and (ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'pacific'))
--or ST_intersects(ST_GEOGPOINT(lon, lat), (select footprint from footprints where region = 'indian')))
and lat < -5
order by timestamp


"""
df = sarm.gbq(q)

# %%
mind = date(2019, 8, 15)
maxd = date(2020, 1, 15)

miny = 54
maxy = 65

minx = -14
maxx = -11

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lon, label=ssvid, s=1)
plt.legend(frameon=False)
plt.title("Longitude of Two MMSI")
plt.xlim(mind, maxd)
# plt.ylim(miny,maxy)
plt.show()

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lat, label=ssvid, s=1)
plt.legend()
plt.title("Latitude of Two MMSI")
plt.xlim(mind, maxd)
# plt.ylim(minx,maxx)

plt.show()
