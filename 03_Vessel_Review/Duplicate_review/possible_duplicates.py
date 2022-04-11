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
# # Review possible duplicates
#
# This notebook contains code used to identify when a vessel might be using two AIS devices, and thus be broadcating two sets of postions with different MMSI at the same time.
#
# The queries access GFW's non-public AIS data.

# %%
import math
# %matplotlib inline
import os

import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
# %%
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
import seaborn as sns

import ais_sar_matching.sar_analysis as sarm

# %load_ext autoreload
# %autoreload 2

# %%
q = """with


AOI as (
SELECT distinct 'indian' as region,  footprint as footprint from  `global-fishing-watch.paper_longline_ais_sar_matching.ksat_detections_ind_v20200110`
union all
select distinct 'pacific' as region,  footprint as footprint from `global-fishing-watch.paper_longline_ais_sar_matching.ksat_detections_fp_v20200117`
),
footprints as (
select ST_UNION_AGG(ST_GEOGFROMTEXT(footprint)) footprint
from AOI),



base_table as(
SELECT a.lat, a.lon, ssvid, vessel_type, hour, _partitiontime date,   distance_from_port_m
  FROM `world-fishing-827.gfw_research_precursors.ais_positions_byssvid_hourly_v20191118` a
join
(select ssvid, vessel_type from `global-fishing-watch.paper_longline_ais_sar_matching.all_detections_and_ais_v20210427`
where not gear)
using(ssvid)
left join
  `pipe_static.distance_from_port_20201105` b
      ON
        CAST( (a.lat*100) AS int64) = CAST( (b.lat*100) AS int64)
        AND CAST((a.lon*100) AS int64) = CAST(b.lon*100 AS int64)
cross join
footprints
where date(_partitiontime) between "2019-09-01" and "2019-12-31"
and st_contains( footprint,st_geogpoint(a.lon,a.lat))
and not interpolated_at_segment_startorend
),

hours_by_ssvid as (
select ssvid,
count(*) hours
from base_table
group by ssvid)


select
a.ssvid,
hour,
date,
a.vessel_type,
c.hours,
b.ssvid,
b.vessel_type,
d.hours,
st_distance(st_geogpoint(a.lon, a.lat), st_geogpoint(b.lon, b.lat)) distance
from
base_table a
join
base_table b
using(date, hour)
join
hours_by_ssvid c
on a.ssvid = c.ssvid
join
hours_by_ssvid d
on b.ssvid = d.ssvid

where a.ssvid = '416623000'
and b.ssvid = '412685210'
order by date, hour
"""
df = sarm.gbq(q)

# %%
df.head()

# %%
sns.histplot(df[df.distance < 500000].distance / 1000)

# %%
from datetime import datetime, timedelta

df["the_date"] = df.apply(lambda x: x.date + timedelta(hours=x.hour), axis=1)

# %%
df.head()

# %%


plt.plot(df.the_date, df.distance / 1000)
plt.ylabel("km apart")
plt.xlabel("date")

# %%
q = """with combined as

(SELECT
lat,
lon,
lag(lat,1) over (order by timestamp) last_lat,
lag(lon,1) over (order by timestamp) last_lon,
lag(timestamp,1) over (order by timestamp) last_timestamp,
timestamp,
speed_knots,
FROM `world-fishing-827.proj_walmart_dark_targets.all_mmsi_positions`
where ssvid in ("416623000","412685210")
and seg_id in (select seg_id from `gfw_research.pipe_v20201001_segs` where good_seg and not overlapping_and_short)
order by timestamp)

select lat, lon, timestamp, speed_knots,
safe_divide ( st_distance(st_geogpoint(last_lon, last_lat), st_geogpoint(lon, lat)) ,
timestamp_diff(timestamp, last_timestamp, second) )*60*60/1852 knots
from combined
where
timestamp_diff(timestamp, last_timestamp, second) > 60

order by timestamp"""

df = sarm.gbq(q)

# %%
df.head()

# %%
d = df  # [(df.timestamp > datetime(2019,9,1))&(df.timestamp < datetime(2019,9,10))]
plt.figure(figsize=(10, 3))
plt.plot(d.timestamp, d.speed_knots)
plt.plot(d.timestamp, d.knots)

# %%
d = df
plt.figure(figsize=(10, 3))
plt.plot(d.timestamp, d.speed_knots)
plt.plot(d.timestamp, d.knots)
# plt.xlim(datetime(2019,9,1), datetime(2019,9,5))

# %%
plt.plot(df.lon, df.lat)

# %%
q = """
select ssvid, lat, lon, timestamp
FROM
`world-fishing-827.proj_walmart_dark_targets.all_mmsi_positions`
where  ssvid in ('631918000', '529813000')
and seg_id in
(select seg_id from `gfw_research.pipe_v20201001_segs` where good_seg and not overlapping_and_short)
order by ssvid, timestamp
"""
df = sarm.gbq(q)


# %%

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lon, label=ssvid, s=1)
plt.legend(frameon=False)
plt.title("Longitude of Two MMSI")
plt.show()

plt.figure(figsize=(20, 3))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    plt.scatter(d.timestamp, d.lat, label=ssvid, s=1)
plt.legend()
plt.title("Latitude of Two MMSI")
plt.show()

# %%
mind = datetime(2019, 9, 1)
maxd = datetime(2019, 11, 1)

miny = -145
maxy = -150

minx = -12
maxx = -8

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
