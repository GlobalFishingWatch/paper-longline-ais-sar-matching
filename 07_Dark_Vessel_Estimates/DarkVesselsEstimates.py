# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
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

# # Estimate the Number of Dark Vessels
#
# 1. Model the relationship between SAR length and AIS length
# 2. Model the recall curve as a proxy for detection rate
# 3. Estimate dark vessels as a function of length, recall, and observations

# +
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyseas.maps as psm
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy import newaxis
from matplotlib.patches import Polygon
plt.rcParams["axes.grid"] = False
import pickle
import warnings

import proplot as pplt
import scipy
from scipy.special import erfinv, gammaln
# %matplotlib inline
from scipy.stats import binom, gaussian_kde, lognorm, poisson
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rc("legend", fontsize="14")
plt.rc("legend", frameon=False)

purple = "#d73b68"
navy = "#204280"
lighter_navy = "#5a7fc4"
orange = "#f68d4b"
gold = "#f8ba47"
green = "#ebe55d"

fishing_color = gold  #'orange'
nonfishing_color = lighter_navy  #'blue'
dark_vessel_color = "#545454"

# %load_ext autoreload
# %autoreload 2

import ais_sar_matching.sar_analysis as sarm
from ais_sar_matching.sar_analysis import (AISPriorCouplingFunction,
                                           BinCouplingFunctionLarge, LComputer)

# %load_ext autoreload
# %autoreload 2


# Important! This is the threshold that we accpet matches
# We found it was between 5e-6 and 1e-4. This is the mean
# of these two values, and in theory we should also test
# how much the model changes based on changing this value
matching_threshold = 2.5e-5
# -

# # Get all detections and non-gear ssvid
#
# Fields:
#  - ssvid: vessel mmsi. If it is null, it is a detection that could not match to anything.
#  - score: matching score for detection to ssvid. Thereshold for accepting a match is between 5*10^-6 and 1*10^-4.
#  - detect_id: unique identifier for SAR detection. null if there is none
#  - gfw_length: If vessel has ssvid, the length from the GFW vessel database.
#  - sar_length: If there is a detection, the lenght reported by SAR
#  - is_fishing: If the vessel is a fishing vessel
#  - likelihod_in: chance (from 0-1) that a vessel is within the scene.
#  - region: either "indian" or "pacific"
#  - within_fooptrint_5km: Is the likely position more than 5 km within the scene?
#  - match_review: is "yes", "ambiguous", "maybe", and "no". Select "yes" for definite matches, which is used for comparing sar length with gfw_length
#
#
# Notes:
#  - a single row can have a ssvid and a detect_id, but represent two different vessels. If the score < 5e-6 it almost definitely represents two distinct vessels (one in AIS that was not detected by SAR and one vessel detected by SAR that was not broadcasting).
#  - there are lots and lots of vessels with a likelihood in <.5
#  - if the sar detection matches to no AIS vessels, it has a score of 0
#  - the same ssvid can appear in multiple scenes
#
# To select all AIS vessels likely in a scene:
#  - df[df.likelihod_in>.5]
#
# To select all AIS to detection pairs that are almost definitely matches:
#  - df[df.match_review == "yes"]
#
#
# Select all SAR detections that represent non-broadcasting vessels, where `matching_threshold` is a value between 5e-6 and 1e-4:
#  - df[(df.score < matching_threshold)&(~df.detect_id.isna())]
#
# Select AIS vessels that likely matched to SAR detections:
# - df[df.score > matching_threshold]

# +

q = """
SELECT
  ssvid,
  score,
  detect_id,
  scene_id,
  gfw_length,
  sar_length,
  is_fishing,
  likelihood_in,
  region,
  within_footprint_5km,
  match_review
FROM
  `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20210427`
WHERE
  vessel_type not in ('gear','duplicate')
  OR detect_id IS not NULL
"""

df = sarm.gbq(q)
# -

# save here to access offline
df.to_csv("all_detections_and_vessels.csv")

df = pd.read_csv("all_detections_and_vessels.csv")

for region in ["indian", "pacific"]:
    print(
        f"sar detections in {region}: {len(df[(df.region==region)&(~df.sar_length.isna())])}"
    )

# # `df_m`
# A dataframe for definite AIS to SAR matches, and used to compute SAR to GFW lengths

df_m = df[df.match_review == "yes"]  # only high confidence, unambiguous matches

# # `df_r`
# Vessels in AIS that definitely appeared in the scene. This is for vessels with a more than 99% chance of appearing in a scene, and it is used to calcluate recall as a function of vessel length.

df_r = df[
    (~df.ssvid.isna()) & (df.likelihood_in > 0.99)
]  # Is a vessel and alsmost definitely in the scene


# ## Calculate detection rate per vessel for large (>60m) and small (<60m) vessels
#
# The 60m cutoff is a bit arbitrary, but there are very few vessels between 60 and 100, so you could choose a higher number and get just about the same results. We also find a roughly linear relationship between detection rate and vessel size
#
# `df_grouped` groups df_r by ssvid, and then calculates `detected_t`, which is the detection rate for that ssvid. An ssvid that appeared in four scenes and was detected once by SAR (as measured by a score > matching_threshold) has detected_t = .25
#
# `df_grouped` is then divided into `df_small` -- vessels under 60 meters -- and `df_large`, vessels > 60 meters. The detection rate for df_large is averaged to get an average detection rate for large vessels.
#
# `detect_t` for vessels under 60 meters is later used to fit a line for vessels under this size.

# +
df_grouped = sarm.get_df_grouped(df_r, matching_threshold)

max_size = 60
# Length greater/smaller than 60 meteres
df_small = df_grouped[df_grouped.gfw_length < max_size]
df_large = df_grouped[df_grouped.gfw_length > max_size]

max_detect_rate = df_large.detected_t.sum() / len(df_large)

print(f"maximum detection rate: {max_detect_rate:.03f}")
print(f"vessels detected >{max_size}m:  {df_large.detected_t.sum():g}")
print(f"total vessels >{max_size}m:     {len(df_large)}")
print(f"total vessels <{max_size}m:     {len(df_small)}")

# -


# ### Calculate binned detection rates.
#
# Calculate the average detection rate for all vessels between 10-20 meters, then between 15-25 meters, and so on (10m wide, step by 5m). This is not used for analysis, but rather to compare with the quantile regression.

# get averaged values every 10 vessels
# this is used to plot the average detection rate.
x_bin5, y_bin5, x_bin5std, y_bin5std = sarm.bin_data(df_r, width=10, median=False)
# # Minimization Approach
# ## Compute the Matrix relating GFW to SAR Lengths
compute_L = LComputer(df_m.gfw_length.values, df_m.sar_length.values)
compute_L.plot_fits()
# compute_L = LComputerQuartiles(df_m.gfw_length.values, df_m.sar_length.values)
# compute_L.plot_fits()
# compute_L = LComputerIQD(df_m.gfw_length.values, df_m.sar_length.values)
# compute_L.plot_fits()
# ##### Infer Lengths

# +

lmin, lmax = 0, 400
n_bins = 400
bins = sarm.compute_bins(lmin, lmax, n_bins)
lengths = sarm.compute_lengths(bins)
compute_L = LComputer(df_m.gfw_length.values, df_m.sar_length.values)
L = compute_L(bins)

## Caculate d, the detection rate
df_grouped = sarm.get_df_grouped(df_r, matching_threshold)
df_small = df_grouped[df_grouped.gfw_length < max_size]
df_large = df_grouped[df_grouped.gfw_length > max_size]
max_detect_rate = df_large.detected_t.sum() / len(df_large)

d = sarm.compute_d(lengths, max_p=max_detect_rate, df_small=df_small)
# max_detect_rate was calculated as detection rate of vessels > 60m


# Coupling Matrix
model_bin_size = 5
assert n_bins % model_bin_size == 0
C = np.zeros([n_bins, n_bins // model_bin_size])
for i in range(n_bins):
    C[i, i // model_bin_size] = 1.0 / model_bin_size


# +
# Select bin coupling function here for everywhere below

bin_coupling_func = BinCouplingFunctionLarge(n_bins, 5, lmin, lmax)
# bin_coupling_func = BinCouplingFunction(n_bins, 5, lmin, lmax)

# +
# vessels not in the sample and likely in the scenes

results_byregion = {}
for region in ["indian", "pacific"]:
    # detections in the region under the scoring threshold
    # non null sar length means it is a sar detection
    o = sarm.compute_o(
        df[
            (df.region == region)
            & (df.score < matching_threshold)
            & (~df.sar_length.isna())
        ],
        lengths,
    )
    results_byregion[region] = sarm.infer_vessels(o, d, L, bin_coupling_func)
    print(f"{region} has {results_byregion[region].x.sum():.0f} dark vessels")
# -


# # Updated Figure 3 and 4


# +
# following is to make a density plot
# drawing on seaborn

from seaborn import kdeplot

dark_distribution = {}
ais_distribution = {}

for region in ["pacific", "indian"]:

    all_sar = results_byregion[region].x

    best_lengths = []
    for j, v in enumerate(all_sar):
        v = int(round(v))
        for z in range(v):
            best_lengths.append(j * model_bin_size + model_bin_size / 2)

    my_data = best_lengths
    my_kde = kdeplot(my_data)
    line = my_kde.lines[0]
    x_all, y_all = line.get_data()
    plt.clf()

    ddark = df[(df.region == region) & (df.score < matching_threshold)]
    my_data = best_lengths
    my_kde = kdeplot(ddark.sar_length)
    line = my_kde.lines[0]
    x_sar, y_sar = line.get_data()
    plt.clf()

    dark_distribution[region] = (
        x_all,
        y_all * len(best_lengths),
        x_sar,
        y_sar * len(ddark),
    )

    d2 = df[~(df.ssvid.isna()) & (df.region == region) & (df.likelihood_in > 0.5)]
    my_kde = kdeplot(list(d2.gfw_length))
    line = my_kde.lines[0]
    x_all, y_all = line.get_data()
    plt.clf()

    dd = d2[d2.score > matching_threshold]
    my_kde = kdeplot(list(dd.sar_length))
    line = my_kde.lines[0]
    x_sar, y_sar = line.get_data()
    plt.clf()

    ais_distribution[region] = x_all, y_all * len(d2), x_sar, y_sar * len(dd)


# +
# all_sar = results_combined.x
all_sar = results_byregion["indian"].x + results_byregion["pacific"].x


best_lengths = []
for j, v in enumerate(all_sar):
    v = int(round(v))
    for z in range(v):
        best_lengths.append(j * model_bin_size + model_bin_size / 2)

my_data = best_lengths
my_kde = kdeplot(my_data)
line = my_kde.lines[0]
x_all, y_all = line.get_data()
plt.clf()

ddark = df[(df.score < matching_threshold)]
my_data = best_lengths
my_kde = kdeplot(ddark.sar_length)
line = my_kde.lines[0]
x_sar, y_sar = line.get_data()
plt.clf()

dark_distribution["all"] = x_all, y_all * len(best_lengths), x_sar, y_sar * len(ddark)


d2 = df[~(df.ssvid.isna()) & (df.likelihood_in > 0.5)]
my_kde = kdeplot(list(d2.gfw_length))
line = my_kde.lines[0]
x_all, y_all = line.get_data()
plt.clf()

dd = d2[d2.score > matching_threshold]
my_kde = kdeplot(list(dd.sar_length))
line = my_kde.lines[0]
x_sar, y_sar = line.get_data()
plt.clf()

ais_distribution["all"] = x_all, y_all * len(d2), x_sar, y_sar * len(dd)
# -


# # Figure 3

# +
array = [[1, 1], [2, 3], [2, 4]]  # the "picture" (1 == subplot A, 2 == subplot B, etc.)

fig, axs = pplt.subplots(
    array,
    figwidth=14,
    sharex=False,
    sharey=False,
    grid=False,
    figheight=14,
    spanx=False,
)

###################################
# Plot actual length vs. SAR length
###################################
pplt.rc["abc.size"] = 14
ax = axs[1]  # first plot

## Plot scatter of fishing and non-fishing vessels
alpha = 1

yf = df_m[df_m.is_fishing].sar_length.values
xf = df_m[df_m.is_fishing].gfw_length.values
ax.scatter(xf, yf, alpha=alpha, color=fishing_color, label="Fishing vessels")

ynf = df_m[~df_m.is_fishing].sar_length.values
xnf = df_m[~df_m.is_fishing].gfw_length.values
ax.scatter(xnf, ynf, alpha=alpha, color=nonfishing_color, label="Non-fishing vessels")


## Fit a quantile regression for .33 and .67 regression

y = df_m.sar_length.values
x = df_m.gfw_length.values
quantiles = [0.333, 0.667]
# quantiles = np.linspace(0,100,101)[1:-1]/100
models = [sarm.fit_line(x, y, q) for q in quantiles]
models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])


def length_model(l):
    """SAR length -> AIS length (Q1, Q2, Q3)"""
    return [get_x(a, b, l) for a, b in zip(models.a, models.b)]


x_ = np.linspace(0, x.max(), 50)
quants = {}
ys = []
quantiles_colors = ["#999999", "#4a4a4a"]
for i in range(models.shape[0]):
    q = models.q[i]
    y_ = sarm.get_y(models.a[i], models.b[i], x_)
    ys.append(y_)
    quants[q] = y_
    ax.plot(
        x_, y_, linestyle="dashed", label="_nolegend_", linewidth=2, color="dimgray"
    )  # quantiles_colors[i])#'dimgray'


ax.set_ylabel("Length estimated from SAR (m)", fontsize=18)
ax.set_xlabel("Vessel length (m)", fontsize=18)
# ax.set_xticks(range(0, 240, 20))
ax.set_xticks(range(0, 250, 50))
ax.set_xlim(0, 240)
ax.set_ylim(0, 355)
leg = ax.legend(
    frameon=False, prop={"size": 16}, ncols=1, loc="upper left"
)  # bbox_to_anchor=(0.3, 1))
ax.tick_params(axis="both", which="major", labelsize=16)
for lh in leg.legendHandles:
    lh.set_alpha(1)

ax.annotate("Quantile 0.67", xy=(190, 272), fontsize=16, rotation=52, weight="bold")

ax.annotate("Quantile 0.33", xy=(185, 198), fontsize=16, rotation=45.5, weight="bold")

# ax.set_title("Vessel Length vs. Length Estimated by SAR", fontsize=18)


##################################
# Recal as function of length
##################################

ax = axs[0]

x_ = np.linspace(0, 65, 50)


def recall_model(l, max_value=max_detect_rate):
    """Length -> probability of detection
    if recall is greater than the max value give
    it the constant we assigned earlier
    """
    recall = models_.a[0] + models_.b[0] * l
    if np.ndim(recall) > 0:
        recall[recall > max_value] = max_value
    elif recall > max_value:
        recall = max_value
    return recall


x = df_small.gfw_length.values
y = df_small.detected_t.values

x_fish = df_grouped[df_grouped.is_fishing == True].gfw_length.values
y_fish = df_grouped[df_grouped.is_fishing == True].detected_t.values

x_nonfish = df_grouped[(df_grouped.is_fishing == False)].gfw_length.values
y_nonfish = df_grouped[(df_grouped.is_fishing == False)].detected_t.values

# plt.scatter(x,y)


ax.scatter(
    x_fish,
    y_fish,
    color=fishing_color,
    s=30,
    alpha=alpha,
    zorder=2,  # , marker="s"
    label="Fishing vessels",
)

ax.scatter(
    x_nonfish,
    y_nonfish,
    color=nonfishing_color,
    s=30,
    alpha=alpha,
    zorder=2,
    label="Non-fishing vessels",
)


## Median line through the detection rate for vessels < 60m
x = df_small.gfw_length.values
y = df_small.detected_t.values
recall_quantiles = [0.5]
models_ = [sarm.fit_line(x, y, q) for q in recall_quantiles]
models_ = pd.DataFrame(models_, columns=["q", "a", "b", "lb", "ub"])
for i in range(models_.shape[0]):
    q = models_.q[i]
    y_ = sarm.get_y(models_.a[i], models_.b[i], x_)
    ax.plot(
        x_[(y_ < max_detect_rate) & (y_ > 0)],  # (y_<max_detect_rate)&(y_>0)
        # makes sure that the line
        y_[(y_ < max_detect_rate) & (y_ > 0)],  # doesn't go above max_detect
        linestyle="dashed",  # rate or below 0
        label=f"Median detection rate, vessels < 60m",
        linewidth=2,
        color="darkgray",  # quantiles_colors[0]
    )


## Plot the detection rate for vessels > 60m as a dashed line
ax.plot(
    [50, 250],
    [max_detect_rate, max_detect_rate],
    linestyle="dotted",
    label="Median detection rate, vessels > 60m",
    color=quantiles_colors[1],
)

## Plot average detection rate for vessels binned at 10m
ax.scatter(
    x_bin5[x_bin5 < 60],
    y_bin5[x_bin5 < 60],
    label="Average detection rate, 10m bins",
    marker="x",
    color="black",
    zorder=3,
)


ax.set_xlabel("Vessel length (m)", fontsize=18)
ax.set_ylabel("SAR detection rate", fontsize=18)
ax.set_xticks(range(0, 240, 20))
# plt.ax_xlim(0, 240)
ax.set_ylim(-0.05, 1.05)
leg = ax.legend(frameon=False, ncols=1, prop={"size": 16}, bbox_to_anchor=(0.5, 0.5))
ax.tick_params(axis="both", which="major", labelsize=16)


##### distributions

region = "all"
x_all, y_all, x_sar, y_sar = ais_distribution[region]
all_v = round(y_all.sum() * (x_all[1] - x_all[0]))
sar_v = round(y_sar.sum() * (x_sar[1] - x_sar[0]))

ax = axs[2]

ax.plot(x_all, y_all, label=f"All, {all_v} vessels", color="#003d9e")
ax.plot(
    x_sar, y_sar, label=f"Detected by SAR, {sar_v} vessels", color="#003d9e", alpha=0.4
)
ax.set_ylim(0, 14)
ax.legend(ncols=1, prop={"size": 16}, bbox_to_anchor=(0.3, 0.3))
ax.tick_params(axis="both", which="major", labelsize=16)


ax.set_ylabel("AIS vessels", fontsize=18)
ax.set_xlim(0, 250)

x_all, y_all, x_sar, y_sar = dark_distribution[region]
all_v = round(y_all.sum() * (x_all[1] - x_all[0]))
sar_v = len(df[(df.score < matching_threshold) & (~df.detect_id.isna())])

ax = axs[3]
ax.plot(x_all, y_all, label=f"All, {all_v} $\pm$ 33 vessels", color="black")
ax.plot(
    x_sar, y_sar, label=f"Detected by SAR, {sar_v} vessels", color="black", alpha=0.4
)
ax.set_ylim(0, 14)
ax.legend(ncols=1, prop={"size": 16}, bbox_to_anchor=(0.3, 0.45))
ax.set_xlim(0, 250)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.set_xlabel("Vessel length (m)", fontsize=18)
ax.set_ylabel("Non-broadcasting vessels", fontsize=18)

ax.text(
    80,
    10,
    "Length distribution for vessels detected\n\
by SAR is length estimated by SAR. Length\n\
distribution for all vessels is true length.",
    fontsize=16,
)


###


axs[1].format(aspect=1)

axs.format(abc="a", fontweight="bold")
pplt.rc["abc.size"] = 18
for ax in axs:
    ax.set_anchor("W")
plt.tight_layout()

# plt.savefig("figure3.png", dpi=300, bbox_inches="tight")
# -

# # Figure 4

# +
### pplt.rc['axes.titlesize'] = 10
plt.rc("legend", fontsize="9")
plt.rcParams["legend.handlelength"] = 1
plt.rcParams["legend.handleheight"] = 1.125
pplt.rc["abc.size"] = 10


d5 = np.zeros(len(all_sar))
for i in range(len(d5)):
    d5[i] = d[i * 5 : i * 5 + 5].mean()

array = [[1, 2], [3, 4]]  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
fig, axs = pplt.subplots(
    ncols=2,
    nrows=4,
    hratios=(6.5, 4.5, 2.5, 2.3),
    figwidth=5.6,
    sharex=True,
    sharey=True,
    space=0,
    figheight=5.6,
    abc="a",
    abcloc="ul",
    grid=False,
    #                         ylabel='vessels',
    xlabel="length, m",
    # the way I do the left labels is funky. I just make a lot of spaces
    # so that the labels line up in the middle
    leftlabels=(
        "",
        "                              Broadcasting AIS",
        "",
        "                Non-broadcasting",
    ),
)

vessel_ylim = (0, 65)
vessel_ylim2 = (0, 45)
dark_ylim = (0, 23)
vessel_xlim = (0, 200)

legend_x = 0.4

for i, region in enumerate(["indian", "pacific"]):

    d2 = df[~(df.ssvid.isna()) & (df.region == region) & (df.likelihood_in > 0.5)]
    d_ais_only = d2[(d2.score < matching_threshold) | (d2.score.isna())]
    d_ais_sar = d2[d2.score > matching_threshold]

    ax = axs[0 + i]

    d2_sar = d2[~d2.sar_length.isna()]
    ax.hist(
        d2_sar.gfw_length,
        bins=80,
        range=(0, 400),
        color="#245abf",
        label=f"AIS and SAR ({len(d2_sar)})",
    )
    ax.text(
        vessel_xlim[1] * 0.4,
        vessel_ylim[1] * 0.2,
        f"{len(d2)} vessels\nbroadcasting AIS",
    )
    ax.set_ylim(vessel_ylim)
    ax.set_xlim(vessel_xlim)

    ax = axs[2 + i]
    # this is just to include the legend
    ax.hist([-10], color="#245abf", label=f"AIS and SAR ({len(d2_sar)})")
    ax.hist(
        d_ais_only.gfw_length,
        bins=80,
        range=(0, 400),
        color="#90a8d4",
        label=f"AIS only ({len(d2)-len(d2_sar)})",
    )

    ax.set_ylim(vessel_ylim2)

    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(legend_x, 0.95))

    all_sar = results_byregion[region].x
    # these give the disributions
    seen = d5 * all_sar
    not_seen = (1 - d5) * all_sar

    num_seen = len(
        df[
            (df.region == region)
            & (df.score < matching_threshold)
            & (~df.detect_id.isna())
        ]
    )
    num_all_sar = all_sar.sum()

    ax = axs[4 + i]

    best_lengths = []
    for j, v in enumerate(seen):
        v = int(round(v))
        for z in range(v):
            best_lengths.append(j * model_bin_size + model_bin_size / 2)

    ax.hist(
        best_lengths,
        bins=80,
        range=(0, 400),
        color="#4a4a4a",
        label=f" SAR only ({num_seen})",
    )
    ax.set_ylim(dark_ylim)

    ax.format(xlabel="Length (m)")

    ax.text(
        vessel_xlim[1] * 0.3,
        dark_ylim[1] * 0.6,
        f"{num_all_sar:.0f} $\pm$ {.175*num_all_sar:.0f} vessels\n not broadcasting AIS",
    )
    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(legend_x, 0.4))

    ax = axs[6 + i]
    best_lengths = []
    for j, v in enumerate(not_seen):
        v = int(round(v))
        for z in range(v):
            best_lengths.append(j * model_bin_size + model_bin_size / 2)

    ax.hist(
        best_lengths,
        bins=80,
        range=(0, 400),
        color="#9c9c9c",
        label=f"Neither SAR nor AIS\n({num_all_sar-num_seen:.0f} $\pm$ {.175*all_sar.sum():.0f})",
    )
    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(legend_x, 1))
    ax.set_ylim(dark_ylim)
axs.format(toplabels=("Indian", "Pacific"))
plt.savefig("figure4.png", dpi=300, bbox_inches="tight")
# -
# ## See How Well it Does
#
# Run simulations

# +
likely_dtcts = df[df.likelihood_in > 0.99]
all_indices = np.arange(len(likely_dtcts))
j = 1
if j != 0 and j % 10 == 0:
    print(".", end="", flush=True)
# Vary the number of samples we take so that we can have varying amounts of
# test data. This allows us to evaluate more of the output space.
sample_size = np.random.randint(len(all_indices) // 4, 4 * len(all_indices) // 5)
sample_size

len(all_indices) // 4, 4 * len(all_indices) // 5, len(all_indices)
# +
# change to True to make your computer do a lot of work
run_simulations = False
n_trials = 10000

if run_simulations:
    act, pred = sarm.run_sims(n_trials=n_trials)
    with open(f"samples_{n_trials}_{matching_threshold}.pickle", "wb") as f:
        pickle.dump({"act": act, "pred": pred}, f)
else:
    print("WARNING: Loading canned data since run_simulations is False!")
    with open(f"samples{n_trials}_{matching_threshold}.pickle", "rb") as f:
        x = pickle.load(f)
    act = x["act"]
    pred = x["pred"]

# +
plt.figure(figsize=(6, 4))
fit = LinearRegression(fit_intercept=False).fit(act[:, None], pred)
x = np.linspace(50, 250)
y = fit.predict(x[:, None])
excess_slope = fit.coef_[0] - 1
# plt.plot(x, y, '--', color="orange")
percent_error = abs(fit.predict(act[:, None]) - pred) / act * 100


plt.title(
    f"Performance of estimating non-broadcasting vessels\n\
mean percent difference +{excess_slope*100:.1f}%, \
mean abs percent error: {percent_error.mean():.1f}%"
)
plt.plot(act, pred, ".", alpha=0.3, markersize=5, markeredgewidth=0)
plt.plot([50, 250], [50, 250], linestyle="--", color="orange")
plt.xlim(40, 250)
plt.ylim(40, 250)
plt.xlabel("actual number of vessels")
plt.ylabel("estimated number of vessels")
plt.savefig("estimation_scatter.png", dpi=300, bbox_inches="tight")

# +
resids = pred - act
resid_width = 50
resid_start, resid_end = 60, 180
locs = np.arange(resid_start, resid_end + 1)
fit_pred = fit.predict(locs[:, None])
w2 = resid_width / 2
cis = []
for lc in locs:
    rs = resids[(act > lc - w2) & (act < lc + w2)]
    cis.append(np.quantile(rs, [0.025, 0.975]))
plt.figure(figsize=(6, 4))
plt.plot(act, pred, ".", markersize=1)
plt.plot(
    locs,
    fit_pred + [x[0] for x in cis],
    "k-",
    linewidth=0.5,
    label="95% confidence bounds",
)
plt.plot(locs, fit_pred + [x[1] for x in cis], "k-", linewidth=0.5)

upper_fit = LinearRegression(fit_intercept=False).fit(
    locs[-50:, None], fit_pred[-50:] + [x[1] for x in cis[-50:]]
)
lower_fit = LinearRegression(fit_intercept=False).fit(
    locs[-50:, None], fit_pred[-50:] + [x[0] for x in cis[-50:]]
)
extended_locs = np.linspace(locs[-1], 250)
plt.plot(
    extended_locs,
    upper_fit.predict(extended_locs[:, None]),
    ":k",
    linewidth=0.5,
    label="extrapolated confidence bounds",
)
plt.plot(extended_locs, lower_fit.predict(extended_locs[:, None]), ":k", linewidth=0.5)
plt.xlim(40, 250)
plt.ylim(40, 250)
plt.xlabel("actual number of vessels")
plt.ylabel("estimated number of vessels")
plt.legend()


# +
def ci_at(x):
    if not (locs[0] < x < locs[-1]):
        raise ValueError("x out of range")
    ndx = np.searchsorted(locs, x)
    return fit_pred[ndx] + cis[ndx]


ndark = 172
print(f"conf int at {ndark}", [round(x) for x in ci_at(ndark)])
# -

plt.hist(pred - act, bins=70)
plt.xlim(-60, 60)
plt.title("distribution of errors on simulations")
plt.ylabel("number of simulations")
plt.xlabel("error (number of vessels)")

# +
resids = pred - act
resid_width = 50
# We want to treat actual vessels as a function of estimated vessels
# to determine the uncertainty in actual vessels

resid_start, resid_end = 15, 210

locs = np.arange(resid_start, resid_end + 1)
fit_pred = fit.predict(locs[:, None])
w2 = resid_width / 2
cis = []
for lc in locs:
    rs = resids[(pred > lc - w2) & (pred < lc + w2)]
    if not rs.any():
        cis.append(np.array([0, 0, 0]))
    else:
        np.quantile(rs, [0.025, 0.5, 0.975])
#     cis.append(np.quantile(rs, [0.025, 0.5, 0.975]))
        cis.append(np.quantile(-rs, [0.025, 0.5, 0.975]))


plt.figure(figsize=(6, 4))
plt.plot(act, pred, ".", markersize=1)
# plt.plot(locs, fit_pred + [x[0] for x in cis], "k-", linewidth=0.5,
#          label='95% confidence bounds')
bot = list(zip(locs, fit_pred + [x[0] for x in cis]))
top = list(zip(locs, fit_pred + [x[2] for x in cis]))
plt.plot(
    locs, fit_pred + [x[1] for x in cis], "k-", linewidth=1.0, label="median prediction"
)
bounds = bot + top[::-1]
ax = plt.gca()
ax.add_patch(Polygon(bounds, facecolor=(0, 0, 0, 0.1)))

plt.xlim(60, 180)
plt.ylim(60, 180)
plt.xlabel("actual number of vessels")
plt.ylabel("estimated number of vessels")
plt.legend()


def ci_at(x):
    if not (locs[0] < x < locs[-1]):
        raise ValueError("x out of range")
    ndx = np.searchsorted(locs, x)
    return fit_pred[ndx] + cis[ndx]


print("conf int at 127", [round(x) for x in ci_at(127)])


# -

print("conf int at 172", [round(x) for x in ci_at(171)])

print("conf int at 191", [round(x) for x in ci_at(191)])

# # What fraction of vessels in each region are fishing that are dark?


# +
regions = ["indian", "pacific"]
dark_vessels_ranges = [[140, 171, 200], [16, 19, 22]]

for region, dark_vessels in zip(regions, dark_vessels_ranges):
    print(region)
    d_ = df[(df.region == region) & (df.likelihood_in > 0.5) * (df.gfw_length < 60)]
    ais_vessels = len(d_)
    fishing_vessels = len(d_[d_.is_fishing])
    frac_fishing = fishing_vessels / ais_vessels

    for dark in dark_vessels:
        dark_fishing = dark * frac_fishing
        per_df = dark_fishing / (fishing_vessels + dark_fishing) * 100
        print(
            f"dark vessels: {dark}, dark fishing: {dark_fishing:.0f}, percent dark fishing: {per_df:.1f}%"
        )
# -


