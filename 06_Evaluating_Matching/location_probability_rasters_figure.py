# %% [markdown]
# # This notebook contains the code to generate the vessel class location probability examples for supplemental figures S1 and S2

import math
# %matplotlib inline
import os

import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
# %%
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
from matplotlib.ticker import FormatStrFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
import cartopy.crs as ccrs


def gbq(q):
    return pd.read_gbq(q, project_id="world-fishing-827")


# %%
q = """with detect_lat_lon as
(select detect_id, detect_lon, detect_lat from `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20210416` where detect_id is not null),
ave_scores as
(select detect_id, ssvid, scene_id, score as score_ave, probability1, probability2 from `world-fishing-827.proj_walmart_dark_targets.matching_v20210421_2_scored` ),
mult_scores as
(select detect_id, ssvid, scene_id, score as score_mult from `world-fishing-827.proj_walmart_dark_targets.matching_v20210421_2_scored_mult` ),
mult_and_ave_scores as
(select * from ave_scores full outer join mult_scores using(ssvid, detect_id, scene_id)),
extrapolated as
(select ssvid, label,
scene_id,
lat1,
lon1,
lat2,
lon2,
likely_lon,
likely_lat,
lat_doppler_offset1,
lon_doppler_offset1,
lat_doppler_offset2,
lon_doppler_offset2,
delta_minutes1,
delta_minutes2,
scale1,
scale2,
max_scale
from `world-fishing-827.proj_walmart_dark_targets.matching_v20210421_1_extrapolated_ais` )
select
scene_id,
ssvid, label,
detect_id,
detect_lon,
detect_lat,
round(st_distance(st_geogpoint(lon1+lon_doppler_offset1, lat1 + lat_doppler_offset1),
            st_geogpoint(detect_lon, detect_lat))/delta_minutes1/1852*60,1) as knots_to,
round(delta_minutes1,1) delta_minutes1,
round(-st_distance(st_geogpoint(lon2+lon_doppler_offset2, lat2 + lat_doppler_offset2),
            st_geogpoint(detect_lon, detect_lat))/delta_minutes2/1852*60,1) as knots_from,
round(delta_minutes2,1) delta_minutes2,
round(st_distance(st_geogpoint(detect_lon, detect_lat),
   st_geogpoint(likely_lon + ifnull(lon_doppler_offset1,0)/2 + ifnull(lon_doppler_offset2,0)/2,
                likely_lat + ifnull(lat_doppler_offset1,0)/2 + ifnull(lat_doppler_offset2,0)/2) )/1000,2) km_likely_to_detect ,
score_mult,
score_ave,
probability1,
probability2,
likely_lon,
likely_lat,
lat1,
lon1,
lat2,
lon2,
lat_doppler_offset1,
lon_doppler_offset1,
lat_doppler_offset2,
lon_doppler_offset2,
scale1,
scale2,
max_scale
from extrapolated
join
mult_and_ave_scores
using(scene_id, ssvid)
join
detect_lat_lon
using(detect_id)
where ssvid in (select ssvid from `world-fishing-827.proj_walmart_dark_targets.all_mmsi_vessel_class` where final_vessel_class not in ("duplicate","gear"))
order by score_ave desc
"""

df_info = gbq(q)


# %% [markdown]
# ## Supplemental vessel location probability raster figures

# %%
def plot_ssvid_scene(ax1, ax2, ax3, ax4, ssvid, scene_id, df):
    plt.rcParams["axes.grid"] = False
    di = df[(df.ssvid == ssvid) & (df.scene_id == scene_id)]

    q = f"""select * from proj_walmart_dark_targets.rasters_single
    where scene_id = '{scene_id}' and ssvid = '{ssvid}' """
    df_r = gbq(q)

    the_max_lat = df_r.detect_lat.max()
    the_max_lon = df_r.detect_lon.max()
    the_min_lat = df_r.detect_lat.min()
    the_min_lon = df_r.detect_lon.min()

    start_lon = di.lon1.values[0]
    end_lon = di.lon2.values[0]
    start_lat = di.lat1.values[0]
    end_lat = di.lat2.values[0]
    likely_lon = di.likely_lon.values[0]
    likely_lat = di.likely_lat.values[0]

    ep = 0.01

    the_max_lat = max([the_max_lat, start_lat + ep, end_lat + ep])
    the_max_lon = max([df_r.detect_lon.max(), start_lon + ep, end_lon + ep])
    the_min_lat = min([df_r.detect_lat.min(), start_lat - ep, end_lat - ep])
    the_min_lon = min([df_r.detect_lon.min(), start_lon - ep, end_lon - ep])

    for delta_minutes in sorted(df_r.delta_minutes.unique(), reverse=True):

        d = df_r[df_r.delta_minutes == delta_minutes]

        min_lon = d.detect_lon.min()
        min_lat = d.detect_lat.min()
        max_lon = d.detect_lon.max()
        max_lat = d.detect_lat.max()

        # get the right scale
        if delta_minutes >= 0:
            scale = di.scale1.values[0]
        else:
            scale = di.scale2.values[0]

        pixels_per_degree = int(round(scale * 111))
        pixels_per_degree_lon = int(
            round(scale * 111 * math.cos((min_lat / 2 + max_lat / 2) * 3.1416 / 180))
        )
        num_lons = int(round((max_lon - min_lon) * pixels_per_degree_lon)) + 1
        num_lats = int(round((max_lat - min_lat) * pixels_per_degree)) + 1
        num_lons, num_lats

        grid = np.zeros(shape=(num_lats, num_lons))

        def fill_grid(r):
            y = int(round((r.detect_lat - min_lat) * pixels_per_degree))
            x = int(round((r.detect_lon - min_lon) * pixels_per_degree_lon))
            grid[y][x] += r.probability

        d.apply(fill_grid, axis=1)

        min_value = 1e-7
        max_value = 1
        #     grid[grid<min_value*100]=0
        norm = mpcolors.LogNorm(vmin=min_value, vmax=max_value)

        if delta_minutes >= 0:
            ax = ax1
        else:
            ax = ax2
        ax.imshow(
            np.flipud(grid),
            norm=norm,
            extent=[min_lon, max_lon, min_lat, max_lat],
            interpolation="nearest",
        )

        ax.set_xlim(the_min_lon, the_max_lon)
        ax.set_ylim(the_min_lat, the_max_lat)
        ax.scatter(
            di.detect_lon, di.detect_lat, color="red", s=7, label="Radarsat2 Detection"
        )
        ax.set_facecolor("#440154FF")
        ax.scatter(
            start_lon,
            start_lat,
            s=7,
            color="orange",
            label="Vessel Position Before Detection",
        )
        ax.scatter(
            end_lon,
            end_lat,
            s=7,
            color="purple",
            label="Vessel Position After Detection",
        )
        if delta_minutes >= 0:
            ax.set_title(f"{delta_minutes:.2f} Minutes\nbefore Image", size=19)
        else:
            ax.set_title(f"{-delta_minutes:.2f} Minutes\nafter Image", size=19)
        ax1.set_xlabel("lon", fontsize=17)
        ax1.set_ylabel("lat", fontsize=17)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.tick_params(axis="both", labelsize=16)

        h, l = ax1.get_legend_handles_labels()
        ax4.legend(
            h, l, loc="upper left", markerscale=2.0, ncol=3, frameon=False, fontsize=14
        )
        ax4.set_facecolor("white")

        ax4.set_xticks([])
        ax4.set_yticks([])

        ax4.grid(False)
        ax4.axis("off")

    # include multiplied raster
    q = f"""select * from proj_walmart_dark_targets.rasters_mult where ssvid = '{ssvid}' and scene_id = '{scene_id}' """
    d = gbq(q)

    min_lon2 = d.detect_lon.min()
    min_lat2 = d.detect_lat.min()
    max_lon2 = d.detect_lon.max()
    max_lat2 = d.detect_lat.max()

    # scale is the larger
    scale = di.max_scale.values[0]

    pixels_per_degree = int(round(scale * 111))
    pixels_per_degree_lon = int(
        round(scale * 111 * math.cos((min_lat / 2 + max_lat / 2) * 3.1416 / 180))
    )
    num_lons = int(round((max_lon2 - min_lon2) * pixels_per_degree_lon)) + 1
    num_lats = int(round((max_lat2 - min_lat2) * pixels_per_degree)) + 1
    num_lons, num_lats

    grid = np.zeros(shape=(num_lats, num_lons))

    def fill_grid(r):
        y = int(round((r.detect_lat - min_lat2) * pixels_per_degree))
        x = int(round((r.detect_lon - min_lon2) * pixels_per_degree_lon))
        grid[y][x] += r.probability

    d.apply(fill_grid, axis=1)

    ax3.scatter(di.detect_lon, di.detect_lat, color="red", s=7)

    min_value = 1e-7
    max_value = 1

    norm = mpcolors.LogNorm(vmin=min_value, vmax=max_value)
    im = ax3.imshow(
        np.flipud(grid),
        norm=norm,
        extent=[min_lon2, max_lon2, min_lat2, max_lat2],
        interpolation="nearest",
    )

    ax3.set_xlim(the_min_lon, the_max_lon)
    ax3.set_ylim(the_min_lat, the_max_lat)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.grid(False)

    ax3.set_facecolor("#440154FF")
    ax3.set_title("Multiplied Probabilities", size=19)

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig2.colorbar(
        im, cax=cax, orientation="vertical", fraction=0.15, aspect=40, pad=0.04
    )

    fontprops = fm.FontProperties(size=14)
    scalebar = AnchoredSizeBar(
        ax3.transData,
        0.09,
        "10 km",
        "lower right",
        pad=0.1,
        color="white",
        frameon=False,
        size_vertical=0,
        fontproperties=fontprops,
    )
    plt.minorticks_off()

    ax3.add_artist(scalebar)


# %% [markdown]
# ### Vessel location probability figure

# %%
fig2 = plt.figure(figsize=(13, 9), constrained_layout=True)
fig2.set_facecolor("white")

gs = fig2.add_gridspec(ncols=3, nrows=2)
minutes = 112
knots = 9

ax1 = fig2.add_subplot(gs[0, 0])
ax2 = fig2.add_subplot(gs[0, 1])
ax3 = fig2.add_subplot(gs[0, 2])
ax4 = fig2.add_subplot(gs[1, 0:])

plot_ssvid_scene(
    ax1,
    ax2,
    ax3,
    ax4,
    "577245000",
    "RS2_20190917_151000_0074_DVWF_HH_SCS_757762_9770_29818070",
    df_info,
)
# fig2.savefig("Fig2_2.png",dpi=300,bbox_inches='tight')

# %% [markdown]
# ## Vessel location raster grid

# %%
df_all = """SELECT
labels,
i,
j,
probability,
prob_contour,
speed_lower,
speed_upper,
minutes_lower,
minutes_upper
FROM
`world-fishing-827.gfw_research_precursors.point_cloud_mirror_nozeroes_contour_v20190502`
where (5 between speed_lower and speed_upper
and -45 between minutes_lower and minutes_upper)
or (3 between speed_lower and speed_upper
and 15 between minutes_lower and minutes_upper)
or (5 between speed_lower and speed_upper
and 30 between minutes_lower and minutes_upper)
or (9 between speed_lower and speed_upper
and 60 between minutes_lower and minutes_upper)
or (13 between speed_lower and speed_upper
and 120 between minutes_lower and minutes_upper)
"""
df2 = gbq(df_all)

# %%
times = [-45, 15, 30, 60, 120]
speeds = [5, 3, 5, 9, 13]

fig3 = plt.figure(figsize=(16, 15))
fig3.set_facecolor("white")

gs1 = fig3.add_gridspec(ncols=6, nrows=5)

labels = [
    "cargo_or_tanker",
    "purse_seines",
    "drifting_longlines",
    "trawlers",
    "tug",
    "other",
]

# times = [15,30]
# speeds = [3,5]

# fig3 = plt.figure(figsize=(6,6))
# fig3.set_facecolor("white")

# gs1 = fig3.add_gridspec(ncols=2, nrows=2)

# labels = ["cargo_or_tanker", "purse_seines"]

for i, (t, s) in enumerate(zip(times, speeds)):
    minutes = t
    knots = s

    for n, l in enumerate(labels):

        d = df2[
            (df2["labels"] == l)
            & (df2["speed_lower"] < s)
            & (df2["speed_upper"] > s)
            & (df2["minutes_lower"] < t)
            & (df2["minutes_upper"] > t)
        ]

        if l == "cargo_or_tanker":
            l = "Cargo or Tanker"

        elif l == "purse_seines":
            l = "Purse Seines"

        elif l == "drifting_longlines":
            l = "Drifting Longlines"

        elif l == "trawlers":
            l = "Trawlers"

        elif l == "tug":
            l = "Tug"

        else:
            l = "Other Vessels"

        grid_d = np.zeros(shape=(1001, 1001))
        grid_p = np.zeros(shape=(1001, 1001))

        def make_grids(row):
            i, j = int(row.i + 500), int(row.j - 500)
            grid_p[i][j] = row.prob_contour
            grid_d[i][j] = row.probability
            if row.i > 0:
                i = int(-row.i + 500)
                grid_p[i][j] = row.prob_contour
                grid_d[i][j] = row.probability

        d.apply(make_grids, axis=1)
        None

        norm = mpcolors.LogNorm(vmin=1e-7, vmax=1)
        ax2 = fig3.add_subplot(gs1[i, n])
        im = ax2.imshow(grid_d, norm=norm)
        ax2.set_facecolor("#440154")
        ax2.set_title(f"{minutes} minutes at\n{knots} knots", size=18)

        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])

fig3.subplots_adjust(right=0.8, hspace=1)
cbar_ax = fig3.add_axes([1, 0.1, 0.03, 0.8])
fig3.colorbar(im, cax=cbar_ax)
cbar_ax.tick_params(labelsize=16)
# cbar_ax.ax.minorticks_on()

fig3.canvas.draw()
plt.tight_layout()

plt.show()


# %%
fig3_final = fig3

# %%
vessels = [
    "Cargo or Tanker",
    "Purse Seines",
    "Drifting Longlines",
    "Trawlers",
    "Tug",
    "Other Vessels",
]

cols = ["{}".format(i) for i in vessels]

pad = 24
for ax, col in zip(fig3_test.axes, cols):
    ax.annotate(
        col,
        xy=(0.5, 1.15),
        xytext=(0, pad),
        xycoords="axes fraction",
        textcoords="offset points",
        size="18",
        weight="bold",
        ha="center",
        va="baseline",
    )


# %%
fig3_final
# fig3_final.savefig("Fig3.png",dpi=300,bbox_inches='tight')
