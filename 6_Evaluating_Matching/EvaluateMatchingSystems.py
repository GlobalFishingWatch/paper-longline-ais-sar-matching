# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## The code in this notebook is used to evaluate our systems for matching SAR detections to AIS and generate supplemental figures 3 and 4

import math
# %matplotlib inline
import os

import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
# +
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rc("legend", fontsize="12")
plt.rc("legend", frameon=False)

purple = "#d73b68"
navy = "#204280"
orange = "#f68d4b"
gold = "#f8ba47"
green = "#ebe55d"

# %load_ext autoreload
# %autoreload 2
# -

score_ave_table = "paper_longline_ais_sar_matching.matching_v20210421_2_scored"
score_mult_table = "paper_longline_ais_sar_matching.matching_v20210421_2_scored_mult"
score_dist_table = "paper_longline_ais_sar_matching.matching_v20210421_2_scored_dist"
detects_reviewed_table = (
    "paper_longline_ais_sar_matching.detections_reviewed_v20210427"
)
score_combined_table = "paper_longline_ais_sar_matching.matching_v20210421_2_scored_combined"
matches_top_table = "paper_longline_ais_sar_matching.matching_v20210421_4_matched_combined"

q = f"""

with ave_mult as (

select a.scene_id, a.ssvid ssvid,
a.detect_id detect_id, a.score score_ave, b.score score_mult, c.score as score_dist,

1000/greatest(ifnull(scale1,0),ifnull(scale2,0)) as min_delta_min,


a.ssvid in
(select ssvid from `global-fishing-watch.paper_longline_ais_sar_matching.all_mmsi_vessel_class`
where final_vessel_class  in ('duplicate','gear') ) as gear_or_duplicate
from `global-fishing-watch.{score_ave_table}` a
join
`global-fishing-watch.{score_mult_table}` b
on a.ssvid = b.ssvid and a.detect_id = b.detect_id
left join
`global-fishing-watch.{score_dist_table}` c
on a.ssvid = c.ssvid and a.detect_id = c.detect_id
),

reviewed_table as
(select
* except(ssvid,score_ave,score_mult,match_category),
 ssvid,
if( b.ssvid is null , "help",  match_category) as match_category

from
(select detect_id,  ssvid from
  `global-fishing-watch.{matches_top_table}` where score > 0) a
left join
`global-fishing-watch.{detects_reviewed_table}` b
using(detect_id, ssvid)
)


select ssvid,a.scene_id, detect_id, score_ave, score_mult,score_combined, score_dist,
gear_or_duplicate,match_category,min_delta_min


from ave_mult a
left join
(select ssvid, score as score_combined, detect_id
from `global-fishing-watch.{score_combined_table}` ) b
using(ssvid,detect_id)
left join
reviewed_table
using(ssvid, detect_id)

"""
df = pd.read_gbq(q)

df.head()

# +
fig1 = fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)

d = df[~df.gear_or_duplicate]

d1 = d[d.match_category == "yes"]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    label="Likely Match",
    color=orange,
    alpha=1,
)

d1 = d[d.match_category.isna()]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Not a Top Match",
    marker="+",
    color="black",
)

d1 = d[d.match_category == "yes_amb"]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Likely Match but Ambiguous",
    color=purple,
)

d1 = d[d.match_category.isin(["maybe", "maybe_amb"])]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Potential Match",
    color=green,
)

d1 = d[d.match_category == "no"]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Unlikely Match",
    color=gold,
)

plt.xlim(-14, 3)
plt.ylim(-11, 3)
plt.xlabel("log10(Muliplied Scores)", fontsize=14)
plt.ylabel("log10(Averaged Scores)", fontsize=14)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.plot([-5.5, -5.5], [-11, 3], "--", color="grey", label="5e-6 Threshold")
plt.plot([-4, -4], [-11, 3], "--", color="black", label="1e-4 Threshold")

# 1:1 line
ax.plot([-11, 3], [-11, 3], color=navy)

# Put a legend to the right of the current axis
ax.legend(bbox_to_anchor=(0.8, 0.5))
plt.title("Averaged vs. Multiplied Scores", fontsize=16)
plt.show()
# fig1.savefig('score_thresh.png', dpi=300, bbox_inches='tight')
# +
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

d = df[(~df.gear_or_duplicate) & (df.min_delta_min < 10)]

d1 = d[d.match_category == "yes"]
plt.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="likely match"
)

d1 = d[d.match_category.isna()]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="not a top match",
    marker="+",
    color="black",
)

d1 = d[d.match_category == "yes_amb"]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="likely match but ambiguous",
)

d1 = d[d.match_category.isin(["maybe", "maybe_amb"])]
plt.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="potential match"
)

d1 = d[d.match_category == "no"]
plt.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="unlikely match"
)

plt.xlim(-14, 3)
plt.ylim(-11, 3)
plt.xlabel("log10(muliplied scores)")
plt.ylabel("log10(averaged scores)")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.plot([-5.5, -5.5], [-11, 3], "--", color="grey", label="5e-6 threshold")
plt.plot([-4, -4], [-11, 3], "--", color="black", label="1e-4 threshold")

# Put a legend to the right of the current axis
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title(
    "Averaged Compared to Multiplied Scores,\nonly Detection to Vessel Pairs within 10 minutes"
)

# +
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

d = df[(~df.gear_or_duplicate) & (df.min_delta_min > 10)]

d1 = d[d.match_category == "yes"]
plt.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="likely match"
)

d1 = d[d.match_category.isna()]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="not a top match",
    marker="+",
    color="black",
)

d1 = d[d.match_category == "yes_amb"]
plt.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="likely match but ambiguous",
)

d1 = d[d.match_category.isin(["maybe", "maybe_amb"])]
plt.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="potential match"
)

d1 = d[d.match_category == "no"]
plt.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="unlikely match"
)

plt.xlim(-14, 3)
plt.ylim(-11, 3)
plt.xlabel("log10(muliplied scores)")
plt.ylabel("log10(averaged scores)")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.plot([-5.5, -5.5], [-11, 3], "--", color="grey", label="5e-6 threshold")
plt.plot([-4, -4], [-11, 3], "--", color="black", label="1e-4 threshold")

# Put a legend to the right of the current axis
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title(
    "Averaged Compared to Multiplied Scores,\nonly Detection to Vessel Pairs greater than 10 minutes"
)

# +
fig = plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

ymin = 0
ymax = 220

plt.plot([-5.5, -5.5], [ymin, 210], "--", color="grey", label="5e-6 threshold")
plt.plot([-4, -4], [ymin, 210], "--", color="black", label="1e-4 threshold")

d = df[~df.gear_or_duplicate]

d1 = d[d.match_category == "yes"]
plt.scatter(
    np.log10(d1.score_combined),
    -(d1.score_dist) / 1000,
    alpha=1,
    label="likely match",
    color="tab:blue",
)

d1 = d[d.match_category == "yes_amb"]
plt.scatter(
    np.log10(d1.score_combined),
    -(d1.score_dist) / 1000,
    alpha=1,
    label="",
    color="tab:blue",
)

d1 = d[d.match_category.isin(["maybe", "maybe_amb"])]
plt.scatter(
    np.log10(d1.score_combined),
    -(d1.score_dist) / 1000,
    alpha=1,
    label="potential match",
    color="tab:gray",
)

d1 = d[d.match_category == "no"]
plt.scatter(
    np.log10(d1.score_combined),
    -(d1.score_dist) / 1000,
    alpha=1,
    label="unlikely match",
    color="tab:red",
)

plt.xlim(-11, 3)
plt.ylim(ymin, ymax)
plt.xlabel("log10(probability), vessel/km2", fontsize=18)
plt.ylabel("Distance between likely location\n and detection (km)", fontsize=18)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc="center left", bbox_to_anchor=(0.65, 0.7), frameon=False, fontsize=15)
plt.title("Distance Metric Compared to Synthetic Probability", fontsize=20)
# plt.savefig("distance_compared_prob.png", dpi=300, bbox_inches = 'tight')

plt.show()
# -

# ## Supplemental figure 3
# ### Comparing averaged to multiplied scores

d = df[(~df.gear_or_duplicate) & (df.min_delta_min < 10)]

# +
paper_fig = plt.figure(figsize=(16, 15))
gs = paper_fig.add_gridspec(2, 2)

d = df[~df.gear_or_duplicate]

# plot 1
ax0 = paper_fig.add_subplot(gs[0, :])

d1 = d[d.match_category == "yes"]
ax0.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    label="Likely Match",
    color="tab:blue",
    alpha=1,
)

d1 = d[d.match_category == "yes_amb"]
ax0.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="", color="tab:blue"
)

d1 = d[d.match_category.isin(["maybe", "maybe_amb"])]
ax0.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Potential Match",
    color="tab:gray",
)

d1 = d[d.match_category == "no"]
ax0.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Unlikely Match",
    color="tab:red",
)

ax0.set_xlim(-14, 3)
ax0.set_ylim(-11, 3)
ax0.set_xlabel("log10(Muliplied Scores)", fontsize=18)
ax0.set_ylabel("log10(Averaged Scores)", fontsize=18)

ax0.plot([-5.5, -5.5], [-11, 3], "--", color="grey", label="5e-6 Threshold")
ax0.plot([-4, -4], [-11, 3], "--", color="black", label="1e-4 Threshold")
ax0.plot([-15, 3], [-5, -5], "--", color="grey", label="5e-6 Threshold")
ax0.plot([-15, 3], [-4, -4], "--", color="black", label="1e-4 Threshold")

# 1:1 line
ax0.plot([-11, 3], [-11, 3], color=navy)

# Put a legend to the right of the current axis
ax0.legend(bbox_to_anchor=(1.05, 0.3), fontsize=15, ncol=2)
plt.title("Averaged vs. Multiplied Scores", fontsize=20)

# plot 2
ax1 = paper_fig.add_subplot(gs[1, 0])

d = df[(~df.gear_or_duplicate) & (df.min_delta_min < 10)]

d1 = d[d.match_category == "yes"]
ax1.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Likely Match",
    color="tab:blue",
)

d1 = d[d.match_category == "yes_amb"]
ax1.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="", color="tab:blue"
)

d1 = d[d.match_category.isin(["maybe", "maybe_amb"])]
ax1.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Potential Match",
    color="tab:gray",
)

d1 = d[d.match_category == "no"]
ax1.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Unlikely Match",
    color="tab:red",
)

ax1.set_xlim(-14, 3)
ax1.set_ylim(-11, 3)
ax1.set_xlabel("log10(Muliplied Scores)", fontsize=18)
ax1.set_ylabel("log10(Averaged Scores)", fontsize=18)

ax1.plot([-5.5, -5.5], [-11, 3], "--", color="grey", label="5e-6 Threshold")
ax1.plot([-4, -4], [-11, 3], "--", color="black", label="1e-4 Threshold")
ax1.plot([-15, 3], [-5, -5], "--", color="grey", label="5e-6 Threshold")
ax1.plot([-15, 3], [-4, -4], "--", color="black", label="1e-4 Threshold")

ax1.set_title(
    "Averaged vs. Multiplied Scores,\nDetection to Vessel Pairs Within 10 Minutes",
    fontsize=20,
)

ax1.plot([-5, 3], [-5, 3], color=navy)

# plot 3
ax2 = paper_fig.add_subplot(gs[1, 1])

d = df[(~df.gear_or_duplicate) & (df.min_delta_min > 10)]

d1 = d[d.match_category == "yes"]
ax2.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="likely Match",
    color="tab:blue",
)

d1 = d[d.match_category == "yes_amb"]
ax2.scatter(
    np.log10(d1.score_mult), np.log10(d1.score_ave), alpha=1, label="", color="tab:blue"
)

d1 = d[d.match_category.isin(["maybe", "maybe_amb"])]
ax2.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Potential Match",
    color="tab:gray",
)

d1 = d[d.match_category == "no"]
ax2.scatter(
    np.log10(d1.score_mult),
    np.log10(d1.score_ave),
    alpha=1,
    label="Unlikely Match",
    color="tab:red",
)

ax2.set_xlim(-14, 3)
ax2.set_ylim(-11, 3)
ax2.set_xlabel("log10(Muliplied Scores)", fontsize=18)
ax2.set_ylabel("", fontsize=18)
ax2.plot([-5.5, -5.5], [-11, 3], "--", color="grey", label="5e-6 Threshold")
ax2.plot([-4, -4], [-11, 3], "--", color="black", label="1e-4 Threshold")
ax2.plot([-15, 3], [-5, -5], "--", color="grey", label="5e-6 Threshold")
ax2.plot([-15, 3], [-4, -4], "--", color="black", label="1e-4 Threshold")

ax2.set_title(
    "Averaged vs. Multiplied Scores,\nDetection to Vessel Pairs Greater than 10 Minutes",
    fontsize=20,
)
ax2.plot([-11, 3], [-11, 3], color=navy)
paper_fig.subplots_adjust(hspace=0.4, wspace=0.4)
plt.show()
# paper_fig.savefig('scores_analysis.png', dpi=300, bbox_inches='tight')
# -




