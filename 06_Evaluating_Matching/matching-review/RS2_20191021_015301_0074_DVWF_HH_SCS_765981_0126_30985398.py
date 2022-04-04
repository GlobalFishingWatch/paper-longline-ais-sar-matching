# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # find `man_adj` matches
#
# - map the `man_adj` detections that did not match, but should have based manual review
# - use FID, scene_id, and visual inspection to find

import os
from datetime import datetime, timedelta

import pandas as pd

# ## select scene

# +
# RS2_20191006_145935_0074_DVWF_HH_SCS_762256_9950_30803268
# RS2_20191011_014440_0074_DVWF_HH_SCS_763423_9997_30848315
# RS2_20191013_145521_0074_DVWF_HH_SCS_764052_0028_30848318
# RS2_20191021_015301_0074_DVWF_HH_SCS_765981_0126_30985398
# RS2_20191028_014848_0074_DVWF_HH_SCS_767720_0207_31056964
# -

scene_id = "RS2_20191021_015301_0074_DVWF_HH_SCS_765981_0126_30985398"

# ### get `unmatched but should've` ssvid from manual review spreadsheet

ssvids_to_find = ["412482958", "416004624"]

# ## set table locations

# +
###INPUT
detection_table = "world-fishing-827.scratch_brian.walmart_ksat_detections_madagascar"
# detection_table_no_gear = 'world-fishing-827.scratch_brian.walmart_ksat_detections_madagascar'
# detection_table_yes_gear = 'world-fishing-827.scratch_brian.walmart_ksat_detections_madagascar'

# ### OUTPUT OLD
# matches_scored = 'scratch_brian.walmart_ksat_madascar_matches'
# matches_ranked = 'scratch_brian.walmart_ksat_madascar_matches_ranked'
# matches_top = 'scratch_brian.walmart_ksat_madascar_matches_top'

### OUTPUT NEW to gear vs no-gear tables
matches_scored_gear = "scratch_brian.walmart_ksat_madascar_matches_v20191213"
matches_ranked_gear = (
    "scratch_brian.walmart_ksat_madascar_matches_ranked_gear_v20191213"
)
matches_top_gear = "scratch_brian.walmart_ksat_madascar_matches_top_gear_v20191213"

### OUTPUT NEW to gear vs no-gear tables
matches_scored_no_gear = "scratch_brian.walmart_ksat_madascar_matches_v20191213"
matches_ranked_no_gear = (
    "scratch_brian.walmart_ksat_madascar_matches_ranked_no_gear_v20191213"
)
matches_top_no_gear = (
    "scratch_brian.walmart_ksat_madascar_matches_top_no_gear_v20191213"
)
# -

# ## get detections, ais near, matches

# +
q = """
select *
from `{detection_table}`
""".format(
    detection_table=detection_table
)

df_ksat_detections = pd.read_gbq(q, project_id="world-fishing-827", dialect="standard")

# +
# print('total detections:', len(df_ksat_detections))

# +
q = """
with
  ais_near_scene as(
    select *
    from `world-fishing-827.scratch_brian.walmart_ksat_ais_near_v20191213`),

  ais_near_scene_w_vessel_type as (
      select ssvid, lat_best, lon_best, lat_center, lon_center, lat_interpolate, lon_interpolate, lat_interpolate2, lon_interpolate2, timeto2, timeto, timestamp, timestamp2, lat, lat2, lon, lon2, speed, speed2, scene_id, within_footprint, best.best_flag, best.best_vessel_class, best.best_length_m, ais_identity.shipname_mostcommon.value as shipname_mostcommon, on_fishing_list_best
      from ais_near_scene
      left join `world-fishing-827.gfw_research.vi_ssvid_v20191218`
      using (ssvid)),

   gear_labels as (
      select ssvid, gear
      from `world-fishing-827.scratch_brian.walmart_ais_near_gear_labels_v20191213`),

   ais_near_scene_w_vessel_type_labelled_gear as (
   select *
   from ais_near_scene_w_vessel_type
   left join gear_labels
   using (ssvid))

select * from ais_near_scene_w_vessel_type_labelled_gear
""".format(
    scene_id=scene_id
)

df_ais_near = pd.read_gbq(q, project_id="world-fishing-827", dialect="standard")

# +
# w/o gear

matches_top = matches_top_no_gear

q = """
select  ssvid, detect_lat, detect_lon, scene_id as id, a_timestamp as start_time, score, detect_id, best.best_vessel_class
from `{matches_top}`
left join `gfw_research.vi_ssvid_v20191218`
using(ssvid)
""".format(
    matches_top=matches_top
)

df_gfw_matches_no_gear = pd.read_gbq(
    q, project_id="world-fishing-827", dialect="standard"
)
# -

import re

import cmocean
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from descartes import PolygonPatch
from geopandas.tools import sjoin
# %matplotlib inline
from matplotlib import colorbar, colors
from matplotlib.collections import LineCollection, PatchCollection
from mpl_toolkits.basemap import Basemap
from shapely import wkt
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.ops import transform

# import shapely.wkt

# #### scene footprint

# +
# get wkt footprint

q = """
select distinct(footprint)
from `{detection_table}`
where scene_id = '{scene_id}'
""".format(
    detection_table=detection_table, scene_id=scene_id
)

df_footprint = pd.read_gbq(q, project_id="world-fishing-827", dialect="standard")
footprint_wkt = wkt.loads(df_footprint.footprint[0])

min_lon, min_lat, max_lon, max_lat = footprint_wkt.bounds
min_lon = min_lon - 0.1
min_lat = min_lat - 0.1
max_lon = max_lon + 0.1
max_lat = max_lat + 0.1
# -

df_footprint.footprint[0]

footprint_wkt = wkt.loads(df_footprint.footprint[0])

min_lon, min_lat, max_lon, max_lat = footprint_wkt.bounds
min_lon, min_lat, max_lon, max_lat

# +
min_lon = min_lon - 0.1
min_lat = min_lat - 0.1
max_lon = max_lon + 0.1
max_lat = max_lat + 0.1

min_lon, min_lat, max_lon, max_lat
# -

# #### EEZs + colormap

# +
# EEZ boundaries & cmap
theShapes = fiona.open(
    "/Users/brianwong/Documents/gfw/data/World_EEZ_v10_20180221/eez_v10.shp"
)

water_color = "#0A1738"
continent_color = "#37496D"
country_line_color = "#222D4B"

# +
# for s in df_ais_near[df_ais_near.scene_id==scene_id].sort_values(by=['ssvid'])[(df_ais_near.within_footprint==True) & (df_ais_near.gear==False)].ssvid.unique():

# for s in ssvids_to_find:

### set map to 'gear' or 'no gear'
df_gfw_matches = df_gfw_matches_no_gear
# df_gfw_matches = df_gfw_matches_gear

### filter tables to current scene
df_gfw_matches_temp = df_gfw_matches[df_gfw_matches.id == scene_id]
df_ksat_detections_temp = df_ksat_detections[df_ksat_detections.scene_id == scene_id]
# df_ais_near_temp = df_ais_near[df_ais_near.scene_id==scene_id]

fig, ax1 = plt.subplots(figsize=(20, 20))

mm = Basemap(
    projection="merc",
    lat_0=min_lat / 2.0 + max_lat / 2.0,
    lon_0=min_lon / 2.0 + max_lon / 2.0,
    resolution="h",
    area_thresh=0.1,
    llcrnrlon=min_lon,
    llcrnrlat=min_lat,
    urcrnrlon=max_lon,
    urcrnrlat=max_lat,
)

mm.drawparallels(
    np.arange(-80.0, 81.0, 0.2),
    color="#ffffff",
    alpha=0.3,
    labels=[True, True, True],
    fontsize=11,
)
mm.drawmeridians(
    np.arange(-180.0, 181.0, 0.2),
    color="#ffffff",
    alpha=0.3,
    labels=[True, True, True],
    fontsize=11,
)

continents = mm.fillcontinents(color=continent_color, lake_color=water_color)

bound = mm.drawmapboundary(fill_color=water_color)

patches = []
for p in theShapes:
    if p["properties"]["MRGID"] in [
        8343,
        8337,
        8348,
    ]:  # 3 values from QGIS attribute properties
        geo = p["geometry"]
        poly = shape(geo)
        if poly.geom_type == "Polygon":
            mpoly = shapely.ops.transform(mm, poly)
            patches.append(PolygonPatch(mpoly))
        elif poly.geom_type == "MultiPolygon":
            for subpoly in poly:
                mpoly = shapely.ops.transform(mm, poly)
                patches.append(PolygonPatch(mpoly))
glaciers = ax1.add_collection(PatchCollection(patches, match_original=True, alpha=0.2))

for index, row in df_footprint.iterrows():
    poly = wkt.loads(row.footprint)
    if poly.geom_type == "Polygon":
        mpoly = shapely.ops.transform(mm, poly)
        patches.append(PolygonPatch(mpoly))
    elif poly.geom_type == "MultiPolygon":
        for subpoly in poly:
            mpoly = shapely.ops.transform(mm, poly)
            patches.append(PolygonPatch(mpoly))
    glaciers = ax1.add_collection(
        PatchCollection(patches, match_original=True, alpha=0.3)
    )

# ksat detections
# df_ksat_matched_unmatched
d = df_ksat_detections_temp
x, y = mm(list(d.lon), list(d.lat))
mm.scatter(x, y, alpha=0.6, s=45, label="KSAT detections", color="red")

### label gfw matches w/ ssvid number
for i in range(len(d)):
    plt.text(
        x[i],
        y[i],
        d.reset_index(drop=True).Name[i],
        va="top",
        family="sans serif",
        weight="light",
        color="white",
        fontsize=14,
    )
###

plt.legend()
plt.tight_layout()
plt.show()
# -

# ## Maps for manual review (w/o gear)

# +
# for s in df_ais_near[df_ais_near.scene_id==scene_id].sort_values(by=['ssvid'])[(df_ais_near.within_footprint==True) & (df_ais_near.gear==False)].ssvid.unique():

for s in ssvids_to_find:

    ### set map to 'gear' or 'no gear'
    df_gfw_matches = df_gfw_matches_no_gear
    # df_gfw_matches = df_gfw_matches_gear

    ### filter tables to current scene
    df_gfw_matches_temp = df_gfw_matches[df_gfw_matches.id == scene_id]
    df_ksat_detections_temp = df_ksat_detections[
        df_ksat_detections.scene_id == scene_id
    ]
    df_ais_near_temp = df_ais_near[df_ais_near.scene_id == scene_id]

    fig, ax1 = plt.subplots(figsize=(20, 20))

    mm = Basemap(
        projection="merc",
        lat_0=min_lat / 2.0 + max_lat / 2.0,
        lon_0=min_lon / 2.0 + max_lon / 2.0,
        resolution="h",
        area_thresh=0.1,
        llcrnrlon=min_lon,
        llcrnrlat=min_lat,
        urcrnrlon=max_lon,
        urcrnrlat=max_lat,
    )

    mm.drawparallels(
        np.arange(-80.0, 81.0, 0.2),
        color="#ffffff",
        alpha=0.3,
        labels=[True, True, True],
        fontsize=11,
    )
    mm.drawmeridians(
        np.arange(-180.0, 181.0, 0.2),
        color="#ffffff",
        alpha=0.3,
        labels=[True, True, True],
        fontsize=11,
    )

    continents = mm.fillcontinents(color=continent_color, lake_color=water_color)

    bound = mm.drawmapboundary(fill_color=water_color)

    patches = []
    for p in theShapes:
        if p["properties"]["MRGID"] in [
            8343,
            8337,
            8348,
        ]:  # 3 values from QGIS attribute properties
            geo = p["geometry"]
            poly = shape(geo)
            if poly.geom_type == "Polygon":
                mpoly = shapely.ops.transform(mm, poly)
                patches.append(PolygonPatch(mpoly))
            elif poly.geom_type == "MultiPolygon":
                for subpoly in poly:
                    mpoly = shapely.ops.transform(mm, poly)
                    patches.append(PolygonPatch(mpoly))
    glaciers = ax1.add_collection(
        PatchCollection(patches, match_original=True, alpha=0.2)
    )

    for index, row in df_footprint.iterrows():
        poly = wkt.loads(row.footprint)
        if poly.geom_type == "Polygon":
            mpoly = shapely.ops.transform(mm, poly)
            patches.append(PolygonPatch(mpoly))
        elif poly.geom_type == "MultiPolygon":
            for subpoly in poly:
                mpoly = shapely.ops.transform(mm, poly)
                patches.append(PolygonPatch(mpoly))
        glaciers = ax1.add_collection(
            PatchCollection(patches, match_original=True, alpha=0.3)
        )

    ### AIS pings before and after
    df = df_ais_near_temp[df_ais_near_temp.ssvid == str(s)]
    lon1, lat1 = mm(df.lon.values, df.lat.values)
    lon2, lat2 = mm(df.lon2.values, df.lat2.values)

    pts = np.c_[lon1, lat1, lon2, lat2].reshape(len(lon1), 2, 2)
    plt.gca().add_collection(LineCollection(pts, color="white", alpha=0.7))

    mm.plot(lon1, lat1, marker=".", ls="", label="AIS position after")
    mm.plot(lon2, lat2, marker=".", ls="", label="AIS position before")

    plt.text(
        lon1,
        lat1,
        df.ssvid.values[0],
        va="bottom",
        family="sans serif",
        weight="light",
        color="gold",
        fontsize=14,
    )
    ###

    # ksat detections
    # df_ksat_matched_unmatched
    d = df_ksat_detections_temp
    x, y = mm(list(d.lon), list(d.lat))
    mm.scatter(x, y, alpha=0.6, s=45, label="KSAT detections", color="red")
    ###

    ### gfw matches
    d = df_gfw_matches_temp.reset_index(drop=True)
    x, y = mm(list(d.detect_lon), list(d.detect_lat))
    mm.scatter(x, y, alpha=1, s=65, label="""gfw matches""", color="springgreen")
    ###

    #     ### label gfw matches w/ ssvid number
    #     for i in range(len(d.ssvid)):
    #         plt.text(x[i], y[i], d.ssvid[i],va="top", family="sans serif", weight="light", color='white', fontsize=14)
    #     ###

    #     ### "df_ais_near best gear position"
    #     d = df_ais_near_temp[df_ais_near_temp.gear == True]
    #     x,y = mm(list(d.lon_best),list(d.lat_best))
    #     mm.scatter(x,y, label=''''best' position gear''', marker='x', s=40, color='yellow', alpha=0.8)
    #     ###

    ### "df_ais_near best real vessel location"
    d = df_ais_near_temp[df_ais_near_temp.gear == False]
    x, y = mm(list(d.lon_best), list(d.lat_best))
    mm.scatter(
        x,
        y,
        label="""'best' position likely vessel""",
        marker="x",
        s=45,
        color="black",
        alpha=0.9,
    )
    ###

    ### "best"
    d = df_ais_near_temp[df_ais_near_temp.ssvid == str(s)]
    x, y = mm(list(d.lon_best), list(d.lat_best))
    mm.scatter(
        x, y, alpha=0.7, s=45, label="""'best' extrapolated position""", color="aqua"
    )
    ###

    ### get detection characteristics for title info
    df_vessel = df_ais_near_temp[df_ais_near_temp.ssvid == s]
    vessel_class = df_vessel["best_vessel_class"].to_string(index=None)
    gear_nogear = df_vessel["gear"].to_string(index=None)
    best_flag = df_vessel["best_flag"].to_string(index=None)

    if s in list(df_gfw_matches_temp.ssvid):
        matched_result = "YES"
    else:
        matched_result = "NO"
    ###

    plt.legend()
    plt.title(
        "ssvid={} | flag={} | vessel class={} | \n gear={} | matched={}".format(
            s, best_flag, vessel_class, gear_nogear, matched_result
        ),
        fontsize=18,
    )
    plt.tight_layout()
    #     plt.savefig('./{}/no-gear/{}.png'.format(scene_id, str(s)), format='png')
    plt.show()
# -
