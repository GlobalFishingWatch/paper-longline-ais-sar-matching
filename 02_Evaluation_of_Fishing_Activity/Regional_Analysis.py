# %% [markdown]
# # This notebook includes the code used to generate figure 1 for Revealing the Global Longline Fleet with Satellite Radar

import math
import string
from itertools import cycle

import cartopy
import cartopy.crs as ccrs
import cmocean
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# %%
import pyseas
import pyseas.cm
import pyseas.maps as psm
import pyseas.maps.rasters
import pyseas.styles
import seaborn as sns
import shapely
from cartopy import config
from matplotlib import colorbar, colors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely import wkt
from six.moves import zip

# psm.use(psm.styles.chart_style)

# GFW Colors
purple = "#d73b68"
navy = "#204280"
orange = "#f68d4b"
gold = "#f8ba47"
green = "#ebe55d"
# %%
# Get the footprints of the acquisitions for Madagascar and French Polynesia
def get_footprint(detection_table):

    q = """
    select distinct(footprint)
    from `{detection_table}`
    """

    df_footprint = pd.read_gbq(
        q.format(detection_table=detection_table),
        project_id="world-fishing-827",
        dialect="standard",
    )

    df_footprint = gpd.GeoDataFrame(df_footprint)
    df_footprint["geometry"] = df_footprint.footprint.apply(wkt.loads)
    overpasses = shapely.ops.cascaded_union(df_footprint.geometry.values)
    return df_footprint, overpasses


# madacascar
detection_table_mad = (
    "world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110"
)
mad_footprints = get_footprint(detection_table_mad)


# french polynesia
detection_table_fp = (
    "world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117"
)
fp_footprints = get_footprint(detection_table_fp)


# %%
# Get the area of the union of each acquisition within each region
def get_area_shape(sh):
    """pass a shapely polygon or multipolygon, get area in square km"""
    b = sh.bounds
    avg_lat = b[1] / 2 + b[3] / 2
    tot_area = sh.area * 111 * 111 * math.cos(avg_lat * 3.1416 / 180)
    return tot_area


print(int(get_area_shape(fp_footprints[1])), "sq km near French Polynesia")
print(int(get_area_shape(mad_footprints[1])), "sq km near Madagascar")

# %%
# Sum of area of individual footprints within each region.
mad_footprints[0]["area"] = mad_footprints[0].geometry.apply(get_area_shape)
fp_footprints[0]["area"] = fp_footprints[0].geometry.apply(get_area_shape)

print(
    "French Polynesia total area sq km:",
    round(fp_footprints[0]["area"].sum(), 2),
    "\nMadacascar total area sq km:",
    round(mad_footprints[0]["area"].sum(), 2),
)

print(
    "\nFrench Polynesia number images:",
    len(fp_footprints[0]),
    "\nMadacascar number of images:",
    len(mad_footprints[0]),
)

# %%
# This query gets longline ais data
q = """

CREATE TEMP FUNCTION hours_diff_ABS(timestamp1 TIMESTAMP,
timestamp2 TIMESTAMP) AS (
#
# Return the absolute value of the diff between the two timestamps
#in hours with microsecond precision
# If either parameter is null, return null
#
ABS(TIMESTAMP_DIFF(timestamp1,
    timestamp2,
    microsecond) / 3600000000.0) );

with
ais_longlines as (
select ssvid,
activity.avg_depth_fishing_m < -200
and best.best_vessel_class = 'drifting_longlines' is_drifting_longline,
on_fishing_list_best
from `world-fishing-827.gfw_research.vi_ssvid_v20210301`
where
activity.offsetting = False
and activity.overlap_hours_multinames < 40
and on_fishing_list_best is not null
and best.best_vessel_class != "gear"
and best.best_vessel_class is not null),

risky_longlines as (
SELECT
  cast(mmsi as string) ssvid
FROM
proj_walmart_dark_targets.risk_predictions_v20201104
join
ais_longlines
on cast(mmsi as string) = ssvid
WHERE
  class = 1
  and year = 2018
),

good_segs as (select seg_id from `gfw_research.pipe_v20201001_segs`
                  where good_seg and not overlapping_and_short ),

ais_positions as
(
select
ssvid,
hours,
nnet_score,
lat,
lon,
is_drifting_longline,
on_fishing_list_best from
`gfw_research.pipe_v20201001`
join
ais_longlines
using(ssvid)
where _partitiontime between timestamp("2019-08-01")
and timestamp("2020-01-31")
and seg_id in (select seg_id from good_segs)
),

all_longline_fishing_andRisky as (
select *,
risky_longlines.ssvid is not null as is_risky
from
ais_positions
left join
risky_longlines
using(ssvid))

select
floor(lat*4) lat_bin,
floor(lon*4) lon_bin,
sum(hours) hours,
sum(if(nnet_score>.5,hours,0)) fishing_hours,
is_drifting_longline,
is_risky,
on_fishing_list_best
from all_longline_fishing_andRisky
group by
lat_bin,
lon_bin,
is_drifting_longline,
on_fishing_list_best,is_risky
"""

df_ais = pd.read_gbq(q, project_id="world-fishing-827")

# %%
# Create longline fishing raster for plotting
ll_df = df_ais[df_ais.is_drifting_longline]
fishing_longlines = pyseas.maps.rasters.df2raster(
    ll_df, "lon_bin", "lat_bin", "fishing_hours", xyscale=4, per_km2=True
)


# %%
# Create a figure of the footprints in one plot for the report
def footprints():
    with pyseas.context(pyseas.styles.dark):
        fig = plt.figure(figsize=(15.5, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        bounds1 = mad_footprints[0].total_bounds
        ax1 = pyseas.maps.create_map(
            gs[0],
            projection="regional.indian",
            extent=([bounds1[0] - 1, bounds1[2] + 1, bounds1[1] - 1, bounds1[3] + 1]),
        )
        pyseas.maps.add_land(ax1)
        pyseas.maps.add_eezs(ax1, edgecolor="white", linewidth=0.4)
        pyseas.maps.add_countries(ax1)
        # pyseas.maps.add_scalebar(ax1)
        ax1.add_geometries(
            mad_footprints[0].geometry.values,
            crs=ccrs.PlateCarree(),
            alpha=0.3,
            edgecolor="k",
        )  # for Lat/Lon data.
        ax1.set_extent([bounds1[0] - 1, bounds1[2] + 1, bounds1[1] - 1, bounds1[3] + 1])

        ax1.add_geometries(
            [mad_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            facecolor="none",
            edgecolor="red",
        )  # for Lat/Lon data.

        plt.title("Madagascar", fontsize=14)

        bounds = fp_footprints[0].total_bounds
        ax2 = pyseas.maps.create_map(
            gs[1],
            projection="regional.south_pacific",
            extent=([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1]),
        )
        pyseas.maps.add_land(ax2)
        pyseas.maps.add_eezs(ax2, edgecolor="white", linewidth=0.4)
        pyseas.maps.add_countries(ax2)
        ax2.add_geometries(
            fp_footprints[0].geometry.values,
            crs=ccrs.PlateCarree(),
            alpha=0.3,
            edgecolor="k",
        )  # for Lat/Lon data.
        ax2.set_extent([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1])

        ax2.add_geometries(
            [fp_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            facecolor="none",
            edgecolor="red",
        )  # for Lat/Lon data.

        plt.title("French Polynesia", fontsize=14)


# %%
# Madagascar Detections and matches including manually identified and eezs
q = """with ksat_matched as
(SELECT ssvid,
detect_id
FROM `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_matches_top_nogear_ind_v20200110` ),

vessels as
(select best.best_vessel_class,
on_fishing_list_best,
best.best_length_m,
registry_info.best_known_flag,
ssvid from `gfw_research.vi_ssvid_v20210301`),

eezs as (
select
distinct ARRAY_TO_STRING(regions.eez, ",") as eez,
floor(lat*4) lat_bin,
floor(lon*4) lon_bin,
from
`world-fishing-827.gfw_research.pipe_v20201001`
where _partitiontime between timestamp("2019-08-01")
and timestamp("2020-01-31")
),

eez_name as (
select cast(eez_id as string) as eez,
reporting_name
from gfw_research.eez_info),

eez_join as (
select * from eezs
left join eez_name
using(eez)
)
select * from (
select * except(ssvid),
if(manual_adj_ssvid is null, ssvid, cast(manual_adj_ssvid as string)) as ssvid,
ssvid is not null or manual_adj_ssvid is not null as matched,
floor(lat*4) lat_bin,
floor(lon*4) lon_bin,
from
proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
left join
ksat_matched
on detect_id = DetectionId
left join
vessels
using(ssvid)
left join `world-fishing-827.proj_walmart_dark_targets.manual_matches_ind_v20200406`
using(DetectionId))
left join eez_join
using(lat_bin, lon_bin)

"""
md_df_detects = pd.read_gbq(q, project_id="world-fishing-827")


# %%
# French Polynesia Detections and matches including manually identified
s = """
with ksat_matched as
(SELECT ssvid,
detect_id
FROM `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_matches_top_nogear_fp_v20200117` ),

vessels as
(select best.best_vessel_class,
on_fishing_list_best,
best.best_length_m,
registry_info.best_known_flag,
ssvid from `gfw_research.vi_ssvid_v20210301`),

eezs as (
select
distinct ARRAY_TO_STRING(regions.eez, ",") as eez,
floor(lat*4) lat_bin,
floor(lon*4) lon_bin,
from
`world-fishing-827.gfw_research.pipe_v20201001`
where _partitiontime between timestamp("2019-08-01")
and timestamp("2020-01-31")
),

eez_name as (
select cast(eez_id as string) as eez,
reporting_name
from gfw_research.eez_info),

eez_join as (
select * from eezs
left join eez_name
using(eez)
)

select * from (
select * except(ssvid),
if(manual_adj_ssvid is null, ssvid, cast(manual_adj_ssvid as string)) as ssvid,
ssvid is not null or manual_adj_ssvid is not null as matched ,
floor(lat*4) lat_bin,
floor(lon*4) lon_bin,
from
proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
left join
ksat_matched
on detect_id = DetectionId
left join
vessels
using(ssvid)
left join `world-fishing-827.proj_walmart_dark_targets.manual_matches_fp_v20200406`
using(DetectionId))
left join eez_join
using(lat_bin, lon_bin)

"""
fp_df_detects = pd.read_gbq(s, project_id="world-fishing-827")


# %% [markdown]
# ## Fig 1

# %%
def label_axes(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (0.9, 0.9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc, xycoords="axes fraction", **kwargs)


# %%
# This is to adjust the plots below to remove the white space between subplots.
# Here we are getting the heights and widths
# of the original data to dynamically set the fig height and weight to account
# for the aspect ratios

with pyseas.context(pyseas.styles.dark):
    bounds1 = mad_footprints[0].total_bounds
    ax1_dim = pyseas.maps.create_map(projection="regional.indian")
    # set the extent of the image
    ax1_dim.set_extent([bounds1[0] - 1, bounds1[2] + 1, bounds1[1] - 1, bounds1[3] + 1])


# get the x and y lims of the image
ax1_dim_left, ax1_dim_right = ax1_dim.get_xlim()
ax1_dim_bottom, ax1_dim_top = ax1_dim.get_ylim()

# get the width and height, which can be used to create the figure aspect
ax1_dim_width = ax1_dim_right - ax1_dim_left
ax1_dim_height = ax1_dim_top - ax1_dim_bottom

# need the same number as number of rows, so duplicating
heights = [ax1_dim_height, ax1_dim_height]
widths = [ax1_dim_width, ax1_dim_width]

# %%
# Plot the figure

# Setting up the figure and gridspec

r = 2  # rows
c = 2  # columns

fig_width = 14  # inches
fig_height = fig_width * sum(heights) / sum(widths)


fig1 = plt.figure(figsize=(fig_width, fig_height))
fig1.set_facecolor("white")
# USE the height ratio's parameter
gs = fig1.add_gridspec(ncols=c, nrows=r, height_ratios=heights)

with np.errstate(invalid="ignore", divide="ignore"):

    with pyseas.context(pyseas.styles.light):
        bounds1 = mad_footprints[0].total_bounds
        ax2 = pyseas.maps.create_map(
            gs[0, 0],
            projection="regional.indian",
            extent=([bounds1[0] - 1, bounds1[2] + 1, bounds1[1] - 1, bounds1[3] + 1]),
        )
        psm.add_land(facecolor="lightgrey", edgecolor="none")
        pyseas.maps.add_eezs(ax2, edgecolor="black", linewidth=0.4)
        pyseas.maps.add_countries(ax2)
        ax2.add_geometries(
            mad_footprints[0].geometry.values,
            crs=ccrs.PlateCarree(),
            alpha=0.2,
            edgecolor="k",
            label="matched to non-fishing vessel",
        )
        ax2.set_extent([bounds1[0] - 1, bounds1[2] + 1, bounds1[1] - 1, bounds1[3] + 1])

        ax2.add_geometries(
            [mad_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            linewidth=2,
            facecolor="none",
            edgecolor="mediumblue",
        )

        AOI_label = mpatches.Rectangle((0, 0), 1, 1, alpha=0.5, edgecolor="mediumblue")
        labels = ["Radarsat2 Scene Footprints"]

        ax2.set_title("Areas imaged by RADARSAT-2", y=1.0, x=0.34, pad=-25, fontsize=20)

        #
        bounds = fp_footprints[0].total_bounds
        ax3 = pyseas.maps.create_map(
            gs[0, 1],
            projection="regional.south_pacific",
            extent=([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1]),
        )
        psm.add_land(facecolor="lightgrey", edgecolor="none")
        pyseas.maps.add_eezs(ax3, edgecolor="black", linewidth=0.4)
        pyseas.maps.add_countries(ax3)
        ax3.add_geometries(
            fp_footprints[0].geometry.values,
            crs=ccrs.PlateCarree(),
            alpha=0.2,
            edgecolor="k",
        )
        ax3.set_extent([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1])

        ax3.add_geometries(
            [fp_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            linewidth=2,
            facecolor="none",
            edgecolor="mediumblue",
        )

        edge = colors.to_rgba("mediumblue", alpha=1)
        face = colors.to_rgba("tab:blue", alpha=0.2)
        AOI_label = mpatches.Rectangle((0, 0), 1, 1, ec=edge, fc=face, linewidth=2)
        labels = ["RADARSAT-2 scene footprints"]

        ax3.legend(
            [AOI_label],
            labels,
            loc="lower left",
            frameon=False,
            fontsize=16,
            handlelength=3,
            handleheight=1.75,
            facecolor="white",
        )

        fig_min_value = 0
        fig_max_value = 1

        projection = "regional.indian"

        ax4 = psm.create_map(gs[1, 0], projection=projection)
        psm.add_land(facecolor="lightgrey", edgecolor="none")
        psm.add_countries()

        psm.add_eezs(ax4, edgecolor="black", linewidth=0.4)
        bounds = mad_footprints[0].total_bounds
        ax4.set_extent([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1])
        ax4.add_geometries(
            [mad_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            linewidth=2,
            facecolor="none",
            edgecolor="mediumblue",
        )

        dotsize = 20

        c1 = "orange"
        c2 = "royalblue"
        c3 = "black"

        d = md_df_detects[md_df_detects.on_fishing_list_best == True]
        ax4.scatter(
            d.lon.values,
            d.lat.values,
            transform=ccrs.PlateCarree(),
            label="Matched to fishing vessel",
            color=c1,
            s=dotsize,
            alpha=0.6,
        )
        d = md_df_detects[(md_df_detects.on_fishing_list_best == False)]
        ax4.scatter(
            d.lon.values,
            d.lat.values,
            transform=ccrs.PlateCarree(),
            label="Matched to non-fishing vessel",
            color=c2,
            s=dotsize,
            alpha=0.6,
        )
        d = md_df_detects[~md_df_detects.matched]
        ax4.scatter(
            d.lon.values,
            d.lat.values,
            transform=ccrs.PlateCarree(),
            color=c3,
            label="Unmatched",
            s=dotsize,
        )
        ax4.set_title("SAR detections", y=1.0, x=0.2, pad=-25, fontsize=20)

        projection2 = "regional.south_pacific"

        ax5 = psm.create_map(gs[1, 1], projection=projection2)
        psm.add_land(facecolor="lightgrey", edgecolor="none")
        psm.add_countries()
        psm.add_eezs(ax5, edgecolor="black", linewidth=0.4)

        bounds = fp_footprints[0].total_bounds
        ax5.set_extent([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1])

        ax5.add_geometries(
            [fp_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            linewidth=2,
            facecolor="none",
            edgecolor="mediumblue",
        )

        d = fp_df_detects[fp_df_detects.on_fishing_list_best == True]
        ax5.scatter(
            d.lon.values,
            d.lat.values,
            transform=ccrs.PlateCarree(),
            label="Matched to Fishing Vessel",
            color=c1,
            s=dotsize,
            alpha=0.6,
        )

        d = fp_df_detects[(fp_df_detects.on_fishing_list_best == False)]
        ax5.scatter(
            d.lon.values,
            d.lat.values,
            transform=ccrs.PlateCarree(),
            label="Matched to Non-Fishing Vessel",
            color=c2,
            s=dotsize,
            alpha=0.6,
        )

        d = fp_df_detects[~fp_df_detects.matched]
        ax5.scatter(
            d.lon.values,
            d.lat.values,
            transform=ccrs.PlateCarree(),
            color=c3,
            label="Unmatched",
            s=dotsize,
        )
        ax5.legend(
            ncol=1,
            loc="lower left",
            frameon=False,
            fontsize=15,
            markerscale=2,
            bbox_to_anchor=(-0.05, -0.03),
        )

        plt.subplots_adjust(wspace=-0.20, hspace=0.05, left=0, right=1, bottom=0, top=1)

        label_axes(
            fig1,
            loc=(0.02, 0.92),
            labels=["b", "c", "d", "e"],
            fontsize=17,
            fontweight="bold",
        )

#         plt.savefig("Fig1.png",dpi=300,bbox_inches='tight')

# %%
fig1 = plt.figure(figsize=(14, 7))
fig1.constrained_layout = True
fig1.set_facecolor("white")
gs = fig1.add_gridspec(ncols=1, nrows=1)

with pyseas.context(pyseas.styles.light):
    grid = np.copy(fishing_longlines)
    grid[grid == 0] = np.nan
    fig_min_value = 0
    fig_max_value = 400
    norm = colors.Normalize(vmin=fig_min_value, vmax=fig_max_value)

    ax1, im1 = psm.plot_raster(
        grid * 1e3,
        gs[0],
        cmap="magma_r",
        norm=norm,
        projection="global.pacific_centered",
    )

    cbar = fig1.colorbar(
        im1,
        ax=ax1,
        orientation="horizontal",
        fraction=0.025,
        aspect=35,
        pad=-0.135,
    )

    cbar.ax.set_title(
        r"Hours per 1000 km$^2$", pad=3, y=1.000001, fontsize=14, loc="left", x=0.2
    )
    cbar.ax.tick_params(labelsize=14)
    pyseas.maps.add_eezs(ax1)
    psm.add_land(facecolor="lightgrey", edgecolor="none")

    ax1.add_geometries(
        [mad_footprints[1]],
        crs=ccrs.PlateCarree(),
        alpha=1,
        facecolor="none",
        edgecolor="mediumblue",
        linewidth=1.5,
    )

    ax1.add_geometries(
        [fp_footprints[1]],
        crs=ccrs.PlateCarree(),
        alpha=1,
        facecolor="none",
        edgecolor="mediumblue",
        linewidth=1.5,
    )

    ax1.set_title(
        "Pelagic Longline Fishing by Vessels with AIS 2019/8 - 2020/1",
        fontsize=19,
        y=1,
        pad=-28,
    )

    plt.text(
        0.245,
        0.78,
        "a",
        fontsize=17,
        transform=plt.gcf().transFigure,
        fontweight="bold",
    )

#     plt.savefig("Fig1_1_worldpanel.png",dpi=300, bbox_inches='tight')

# %%
