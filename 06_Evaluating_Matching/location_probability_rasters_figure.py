# %% [markdown]
# # This notebook contains the code to generate the vessel class location probability examples for supplemental figures S1 and S2
#
# %%
import math
# %matplotlib inline
import os
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
import cartopy.crs as ccrs

import ais_sar_matching.sar_analysis as sarm

# %load_ext autoreload
# %autoreload 2

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

df_info = sarm.gbq(q)

# %% [markdown]
# ## Supplemental vessel location probability raster figures

# %%
plt.rcParams["axes.grid"] = False
fig2 = plt.figure(figsize=(13, 9), constrained_layout=True)
fig2.set_facecolor("white")

gs = fig2.add_gridspec(ncols=3, nrows=2)
minutes = 112
knots = 9

ax1 = fig2.add_subplot(gs[0, 0])
ax2 = fig2.add_subplot(gs[0, 1])
ax3 = fig2.add_subplot(gs[0, 2])
ax4 = fig2.add_subplot(gs[1, 0:])

sarm.plot_ssvid_scene(
    fig2,
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
df2 = sarm.gbq(df_all)

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
for ax, col in zip(fig3_final.axes, cols):
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

# %%
