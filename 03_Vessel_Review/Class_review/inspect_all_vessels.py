# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Review Vessel Types
#
# Feb 20, 2021
#
# It matters a lot which vessels are gear and which are not for our analysis -- so we have to inspect all the vessels to make sure. Also, it woudl be good to know how many fishing vessels there actually are and how many drifting longlines.
#
# Key findings: A number of mmsi vessel classes have been updated, and a

import os

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

import ais_sar_matching.sar_analysis as sarm

# %matplotlib inline

# %load_ext autoreload
# %autoreload 2
# -

# # Things that are Gear

# +
q = """
with good_segs as (
        select seg_id
        from `gfw_research.pipe_v20201001_segs`
        where good_seg)


select ssvid, lat, lon, timestamp, speed_knots from
proj_walmart_dark_targets.all_mmsi_positions
join
good_segs
using(seg_id)
where ssvid not  in (
select ssvid from
`world-fishing-827.proj_walmart_dark_targets.all_mmsi_vessel_class`
where final_vessel_class = 'gear' )


"""

df = sarm.gbq(q)

# +
q = """select ssvid, ifnull(best.best_vessel_class, "none") vessel_class, ais_identity.shipname_mostcommon.value	as ship_name
from `world-fishing-827.gfw_research.vi_ssvid_v20210202`
where ssvid in (select distinct ssvid from proj_walmart_dark_targets.all_mmsi_positions)"""

df_i = sarm.gbq(q)
# -

df = df.merge(df_i, on="ssvid", how="left").fillna("None")

df

sarm.plot_track_speed(df)


# Pete's Review
#
# 0 – high spoofing
# 412000000 – spoofing, no class, zoom in
# 100900000 – not sure what this is doing, class non fishing. May be longliner zoom in
# 200000000 – spoofing zoom in
# 999000009 – possible gear?
# 416005540 – not sure – check tracks in AOI
# 81562598 – gear
# 416000001 – none type, appears to be longliner, spoofing. Zoom in
# 98345251 – likely gear
# 81060166 – likely gear
# 81060610 – likely gear
# 985380199 – not sure, possible gear, zoom in
# 416888888 – spoofing vessel
# 413111113 – spoofing, possible gear, zoom in
# 664050000 – irregular tracks, no gear type, spoofing possible, look at aoi
# 898537605 – gear
# 228368700 – possible longliner zoom in to aoi
# 416888888 – spoofing, no type, zoom in
# 81060633 - gear
# 98344815 – gear
# 98345253 – gear
# 412549054 – no class, zoom in on aoi. Appears to be transiting through aoi
# 9203941 – gear
# 413699230 – listed as passenger, but appears to be gear
# 81561985 – gear
# 9117152 – gear
# 546012200 – listed as non-fishing, but zoom into aoi. May be a longliner
# 412482984 – spoofing. Zoom into aoi. Listed as a longline. Probably a longline
# 81060164 – gear
# 985380200 – likely gear. Listed as passenger
# 81060609 – gear
# 71060011 – gear
# 98347056 – listed as gear, but check aoi. Appears to be moving too fast. Maybe on ship
# 224883000 – possible spoofing. Zoom into aoi. Maybe ditch
# 100000324 – spoofing vessel. Listed as fishing, but looks like it may be gear at aoi. Zoom in and check
# 71060021 – gear
# 999001265 – listed as gear. Looks to either be on board or not gear
# 664130568 – no class. Not sure – not gear too fast. Not fishing. Very few points in track
# 994018093 – listed as gear. Not gear. Longliner?
# 98347499 – listed as gear, but moving pretty quick. Maybe on board? Few points though.
# 81562698 – likely gear, although some faster points
# 981000263 – gear, but looks to be transporting at times. Pretty quick points
# 990115039 – gear being transported at times
# 98347069 – gear being transported at times
# 81562347 – likely gear, but moving fairly quick at times
# 9203975 – gear
# 81561988- gear
# 647791229 – shotgun pattern – gear, tug?
# 318000009 – listed as gear. Gear pattern, but fast… on-board at times?
# 98347370 – gear
# 71060019 – gear
# 577259000 – gear?? Listed as drifting longline. Moving farily quick but not sure about track
# 990119002 – appears to be gear travelling on a longliner, deployed at times
# 647791203 – shotgun pattern – tug?
# 81562589 – gear
# 81561998 – gear
# 81562510 – gear
# 81060165 – gear
# 898517359 – gear
# 71060022 – gear
# 98345278 – gear
# 81561979 – gear
# 81060167 – gear
# 98514469 – gear
# 227151170 – very few points <5… gear?
# 81561991 – gear
# 836777014 – gear
# 98511236 – gear
# 91060099 – gear
# 9115865 – gear
# 81560368 – gear
# 367753890 – likely gear, listed as passenger
# 416004865 – gear
# 91060100 - geara
#

# +
# get tracks within footprint bounding box and plot to make final call
# -

# Pete's Review
pete_updates = {
    "0": "spoofing",
    "412000000": "spoofing",
    "200000000": "spoofing",
    "999000009": "gear",
    "81562598": "gear",
    "416000001": "spoofing",
    "98345251": "gear",
    "81060166": "gear",
    "81060610": "gear",
    "416888888": "spoofing",
    "413111113": "spoofing",
    "664050000": "spoofing",
    "898537605": "gear",
    "416888888": "spoofing",
    "81060633": "gear",
    "98344815": "gear",
    "98345253": "gear",
    "9203941": "gear",
    "413699230": "gear",
    "81561985": "gear",
    "9117152": "gear",
    "412482984": "spoofing",
    "81060164": "gear",
    "985380200": "gear",
    "81060609": "gear",
    "71060011": "gear",
    "224883000": "spoofing",
    "100000324": "spoofing",
    "71060021": "gear",
    "81562698": "gear",
    "981000263": "gear",
    "990115039": "gear",
    "98347069": "gear",
    "81562347": "gear",
    "9203975": "gear",
    "81561988": "gear",
    "98347370": "gear",
    "71060019": "gear",
    "577259000": "gear",
    "81562589": "gear",
    "81561998": "gear",
    "81562510": "gear",
    "81060165": "gear",
    "898517359": "gear",
    "71060022": "gear",
    "98345278": "gear",
    "81561979": "gear",
    "81060167": "gear",
    "98514469": "gear",
    "227151170": "gear",
    "81561991": "gear",
    "836777014": "gear",
    "98511236": "gear",
    "91060099": "gear",
    "9115865": "gear",
    "81560368": "gear",
    "367753890": "gear",
    "416004865": "gear",
    "91060100": "gear",
}

# +
# further review these vessels tracks within AOIs

mmsi = [
    "100900000",
    "416000001",
    "416005540",
    "985380199",
    "413111113",
    "664050000",
    "228368700",
    "416888888",
    "412549054",
    "546012200",
    "412482984",
    "98347056",
    "224883000",
    "100000324",
    "994018093",
    "412000000",
    "200000000",
    "994161211",
    "416000001",
    "664050000",
    "412333331",
    "577259000",
    "81562510",
]
mmsi = set(mmsi)
mmsi = list(mmsi)
# -

# David's review:
#
#
# 412000000 - highly spoofing vessel... should map in the regions of interest
#
# 413111113 - is most likely gear... it's name is percentages
#
# 0 - who uses 0 as the ship name!!!
#
# 100900000 -- highly spoofing, looks like likely drifting longline for the area of interest
#
# 100000324 -- looks like gear... and it is gear, even though it's best class is fishing. It's inferred value is gear.
#
# 200000000 -- highly spoofing, look at it in the region of interest
#
#
# 81562591 -- gear
#
# 994161211 -- listed as gear, but map it out in the region of interest to make sure because too zoomed out to tell
#
# 224883000 -- a purse serene that turns off its AIS a lot
#
# 416000001 -- highly spoofing vessel, look at tracks in region of interest (Indian Ocean) -- looks like a drifting longline there
#
# 664050000 -- highly spoofing vessel, look at tracks in region of interest (Indian Ocean)
#
# 81560367 -- gear based on behavior
#
# 412333331 - highly spoofing vessel, look at tracks in region of interest -- looks like a longline
#
# 577259000 -- map more of this vessels' activity, because it has very little and hard to tell
#
# 81060610 -- gear, based on behavior and name
#
# 98345253 -- gear, based on name and behavior
#
# 999000009 -- spoofing, is gear in our region (Indian Ocean)
#
# 81562598 - gear, behavior and name
#
# 9203975 -- gear, based on name and behavior
#
# 81060166 - gear
#
# 98347370 - gear
#
# 71060019 - gear
#
# 81562404 - likely gear
#
# 81060609 - likely gear
#
# 98344815 - gear
#
# 71060011 - likely gear (very sparse tracks, and name JIN HUNG NO.308-7)
#
# 81060167 - likely gear
#
# 81561985 - likely gear
#
# 413699230 - gear! (It is labeled as passenger, but it is gear based on name and behavior)
#
# 81562698 - gear
#
# 81561998 - gear
#
# 81060164 - gear
#
# 898537605 - likely gear, very likely based on behavior, but no name
#
# 98345251 - gear based on name and behavior
#
# 81560368 - gear
#
# 81562347 - likely gear
#
# 81060165 - gear
#
# 81561979 - gear
#
# 416004865 - extremely sparse tracks, but believed to be drifting longline. Look at a longer track of this vessel
#
# 9203941 - gear (% in name, sparse tracks)
#
# 9115865 - unknown
#
# 81060633 - gear
#
# 81562510 - likely gear -- make a longer track for review
#
# 71060021 - gear
#
# 81561988 - gear
#
# 836777014 - likely gear
#
# 98511236 - gear
#
# 898517359 - gear
#
# 98345278 - gear
#
# 91060100 - gear
#
# 91060099 - gear
#

vessel_updates = {
    "412000000": "spoofing",  #  highly spoofing vessel... should map in the regions of interest
    "413111113": "gear",  #  is most likely gear... it's name is percentages
    "0": "spoofing",  #  who uses 0 as the ship name!!!
    "100900000": "spoofing",  # - highly spoofing, looks like likely drifting longline for the area of interest
    "100000324": "gear",  # - looks like gear... and it is gear, even though it's best class is fishing. It's inferred value is gear.
    "200000000": "spoofing",  # - highly spoofing, look at it in the region of interest
    "81562591": "gear",  # - gear
    "994161211": "gear",  # - listed as gear, but map it out in the region of interest to make sure because too zoomed out to tell
    "224883000": "tuna_purse_seine",  # - a purse serene that turns off its AIS a lot
    "416000001": "spoofing",  # - highly spoofing vessel, look at tracks in region of interest (Indian Ocean)": "", # - looks like a drifting longline there
    "664050000": "spoofing",  # - highly spoofing vessel, look at tracks in region of interest (Indian Ocean)
    "81560367": "gear",  # - gear based on behavior
    "412333331": "spoofing",  #  highly spoofing vessel, look at tracks in region of interest": "", # - looks like a longline
    "577259000": "map",  # - map more of this vessels' activity, because it has very little and hard to tell
    "81060610": "gear",  # - gear, based on behavior and name
    "98345253": "gear",  # - gear, based on name and behavior
    "999000009": "gear",  # - spoofing, is gear in our region (Indian Ocean)
    "81562598": "gear",  #  gear, behavior and name
    "9203975": "gear",  # - gear, based on name and behavior
    "81060166": "gear",  #  gear
    "98347370": "gear",  #  gear
    "71060019": "gear",  #  gear
    "81562404": "gear",  #  likely gear
    "81060609": "gear",  #  likely gear
    "98344815": "gear",  #  gear
    "71060011": "gear",  #  likely gear (very sparse tracks, and name JIN HUNG NO.308-7)
    "81060167": "gear",  #  likely gear
    "81561985": "gear",  #  likely gear
    "413699230": "gear",  #  gear! (It is labeled as passenger, but it is gear based on name and behavior)
    "81562698": "gear",  #  gear
    "81561998": "gear",  #  gear
    "81060164": "gear",  #  gear
    "898537605": "gear",  #  likely gear, very likely based on behavior, but no name
    "98345251": "gear",  #  gear based on name and behavior
    "81560368": "gear",  #  gear
    "81562347": "gear",  #  likely gear
    "81060165": "gear",  #  gear
    "81561979": "gear",  #  gear
    "416004865": "drifting_longlines",  #  extremely sparse tracks, but believed to be drifting longline. Look at a longer track of this vessel
    "9203941": "gear",  #  gear (% in name, sparse tracks)
    "9115865": "gear",  #  unknown
    "81060633": "gear",  #  gear
    "81562510": "gear",  #  likely gear": "", # - make a longer track for review
    "71060021": "gear",  #  gear
    "81561988": "gear",  #  gear
    "836777014": "gear",  #  likely gear
    "98511236": "gear",  #  gear
    "898517359": "gear",  #  gear
    "98345278": "gear",  #  gear
    "91060100": "gear",  #  gear
    "91060099": "gear",
}  #  gear}

# Plot the ones that we said we should look at within the AOIs

"416002477",
"81562154",
"816666014",
"111113278",
"98346881",
"98346237",
"664130568"


# +
closer_look = """with good_segs as (
        select seg_id
        from `gfw_research.pipe_v20201001_segs`
        where good_seg),
tracks as (
select
ssvid,
seg_id,
lat,
lon,
timestamp,
speed_knots
from
proj_walmart_dark_targets.all_mmsi_positions
join
good_segs
using(seg_id)
where ssvid in
 ('2020202', '1056964608', '412549103', '168888801', '416005715', '4168888'
 )),

all_footprints as (
select
  footprint
from
  proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 select
   footprint
 from
   proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
)
select
  ssvid,
  seg_id,
  lat,
  lon,
  timestamp,
  speed_knots
from
  tracks
cross join
   all_footprints
  where
   st_distance(ST_GEOGFROMTEXT(footprint), st_geogpoint(lon, lat))< 300*1000


"""

AOI_tracks = sarm.gbq(closer_look)
# -

AOI_tracks = AOI_tracks.merge(df_i, on="ssvid", how="left").fillna("None")

AOI_tracks.head()

AOI_tracks.ssvid.nunique()

sarm.plot_track_speed(AOI_tracks)

# +
closer_look = """with good_segs as (
        select seg_id
        from `gfw_research.pipe_v20201001_segs`
        where good_seg),
tracks as (
select
ssvid,
seg_id,
lat,
lon,
timestamp,
speed_knots
from
proj_walmart_dark_targets.all_mmsi_positions
join
good_segs
using(seg_id)
where ssvid in
 ('416002477','600000001','2020202','100000324'
 )),

all_footprints as (
select
  footprint
from
  proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 select
   footprint
 from
   proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
)
select
  ssvid,
  seg_id,
  lat,
  lon,
  timestamp,
  speed_knots
from
  tracks
cross join
   all_footprints
  where
   st_distance(ST_GEOGFROMTEXT(footprint), st_geogpoint(lon, lat))< 300*1000


"""

AOI_tracks = sarm.gbq(closer_look)
# -

AOI_tracks = AOI_tracks.merge(df_i, on="ssvid", how="left").fillna("None")
sarm.plot_track_speed(AOI_tracks)

# +
closer_look = """with good_segs as (
        select seg_id
        from `gfw_research.pipe_v20201001_segs`
        where good_seg),
tracks as (
select
ssvid,
seg_id,
lat,
lon,
timestamp,
speed_knots
from
proj_walmart_dark_targets.all_mmsi_positions
join
good_segs
using(seg_id)
where ssvid in
 ('200000000'
 )),

all_footprints as (
select
  footprint
from
  proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 select
   footprint
 from
   proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
)
select
  ssvid,
  seg_id,
  lat,
  lon,
  timestamp,
  speed_knots
from
  tracks
cross join
   all_footprints
  where
   st_distance(ST_GEOGFROMTEXT(footprint), st_geogpoint(lon, lat))< 300*1000


"""

AOI_tracks = sarm.gbq(closer_look)
# -

AOI_tracks = AOI_tracks.merge(df_i, on="ssvid", how="left").fillna("None")
sarm.plot_track_speed(AOI_tracks)


# +
# update list from further review

update_dict = {
    "81060418": "gear",
    "898537606": "gear",
    "81562219": "gear",
    "81562508": "gear",
    "81561999": "gear",
    "81562003": "gear",
    "647591304": "gear",
    "932555512": "gear",
    "81562156": "gear",
    "932555510": "gear",
    "81060608": "gear",
    "81562155": "gear",
    "816666012": "gear",
    "836777011": "gear",
    "81060606": "gear",
    "416002477": "gear",
    "81562154": "gear",
    "816666014": "gear",
    "111113278": "gear",
    "98346881": "gear",
    "98346237": "gear",
    "664130568": "gear",
    "994018093": "drifting_longlines",
    "0": "depends",
    "412000000": "depends",
    "412482984": "drifting_longline",
    "100000324": "gear",
    "416888888": "drifting_longlines",
    "200000000": "spoofing",
    "664050000": "drifting_longlines",
    "416000001": "drifting_longlines",
    "228368700": "non_fishing",
    "413111113": "gear",
    "81562510": "gear",
    "412549054": "non_fishing",
    "577259000": "gear",
    "224883000": "non_fishing",
    "416888888": "drifting_longlines",
    "200000000": "drifting_longlines",
    "100900000": "drifting_longlines",
    "412333331": "drifting_longlines",
}


# -

# Merge Pete's, David's, and update list into final list.


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


final_list = Merge(pete_updates, vessel_updates)

final_list = Merge(final_list, update_dict)

final_list

for i in final_list:
    if final_list[i] == "spoofing":
        print(i)

final_class_list = (
    pd.DataFrame.from_dict(final_list, orient="index")
    .reset_index()
    .rename(columns={"index": "ssvid", 0: "best_vessel_class"})
)

# +
# final_class_list.to_gbq('scratch_pete.walmart_darkTargets_ssvid_Class_final', project_id='world-fishing-827', if_exists='replace')

# +
final = """with

old_gear as (select ssvid, 'gear' as final_vessel_class
from `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221`
where gear
and ssvid not in ('100900000',
'228368700',
'412549103',
'416005715',
'4168888'))

select distinct * from (
select * from proj_walmart_dark_targets.all_mmsi_vessel_class
union all
select * from old_gear)"""

f1 = sarm.gbq(final)
# -

len(f1)

# +
# f1.to_gbq('proj_walmart_dark_targets.all_mmsi_vessel_class', project_id='world-fishing-827', if_exists='replace')
# -

# # Vessels of Concern

q = """select ssvid, best.best_vessel_class from `world-fishing-827.gfw_research.vi_ssvid_v20210301` where
best.best_vessel_class != "gear" and ssvid in (
select ssvid from proj_walmart_dark_targets.all_mmsi_vessel_class where final_vessel_class = 'gear')
"""
df_conern = sarm.gbq(q)

df_conern

# ssvid	best_vessel_class
# 985380200	passenger --> passenger, as it is operating only close to shore (very strange that it operates in two regions, though)
# 416002477	drifting_longlines --> map in region -- is drifting longline!!!
# 413111113	non_fishing -- > yes is gear
# 1004	squid_jigger
# 81562404	drifting_longlines --> yes, gear
# 600000001	non_fishing --> map in region...
# 999000009	non_fishing --> gear
# 413699230	passenger --> gear
# 2020202	non_fishing --> map in region -->gear
# 200000003	squid_jigger --> gear
# 168888801	fishing --> gear
# 100000324	fishing -- map in region
# 367753890	passenger --> passenger
# 1056964608	squid_jigger --> gear
#
# changes:
#
# '985380200':'passenger'
# '416002477':'drifting_longlines'
# '367753890':'passenger'
#

df = sarm.gbq("select * from proj_walmart_dark_targets.all_mmsi_vessel_class")
df.head()

df.at[df.ssvid == "985380200", "final_vessel_class"] = "passenger"
df.at[df.ssvid == "416002477", "final_vessel_class"] = "drifting_longlines"
df.at[df.ssvid == "367753890", "final_vessel_class"] = "passenger"

df[df.ssvid == "367753890"]

# +
# df[["ssvid", "final_vessel_class"]].to_gbq(
#     "proj_walmart_dark_targets.all_mmsi_vessel_class",
#     project_id="world-fishing-827",
#     if_exists="replace",
# )
