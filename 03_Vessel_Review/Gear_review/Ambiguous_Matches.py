# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Review ambigous matches - AIS that could have matched to multiple SAR detections, or SAR to multiple vessels broadcasting AIS.

from datetime import datetime, timedelta

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
# %%
from pyseas.contrib import plot_tracks

# %matplotlib inline


def gbq(q):
    return pd.read_gbq(q, project_id="world-fishing-827")


# %%
def plot_track_speed(df):
    """
    Function to plot vessels track points colored by speed,
    with a histogram of speed
    """

    with pyseas.context(styles.light):
        for ssvid in df.ssvid.unique():

            ssvid_df = df[df.ssvid == ssvid]

            year = np.sort(ssvid_df.year.unique())

            for i in year:
                d = ssvid_df[ssvid_df["year"] == i]

                fig = plt.figure(
                    figsize=(10, 14),
                )
                gs = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
                projinfo = plot_tracks.find_projection(df.lon, df.lat)

                ax = maps.create_map(gs[0])
                ax.set_global()
                maps.add_land()
                maps.add_countries()
                maps.add_eezs()
                maps.add_gridlines()
                maps.add_gridlabels()

                cm = plt.cm.get_cmap("RdYlBu_r")
                z = np.array(d.speed_knots)

                normalize = mcol.Normalize(vmin=0, vmax=10, clip=True)
                zero = ax.scatter(
                    d.lon.values,
                    d.lat.values,
                    c=z,
                    cmap=cm,
                    norm=normalize,
                    alpha=0.7,
                    transform=maps.identity,
                )

                cbar0 = plt.colorbar(zero)
                cbar0.set_label("Speed", rotation=270)

                ax1 = maps.create_map(
                    gs[1], projection=projinfo.projection, extent=projinfo.extent
                )
                maps.add_land()
                maps.add_countries()
                maps.add_eezs()
                maps.add_gridlines()
                maps.add_gridlabels()

                one = ax1.scatter(
                    d.lon.values,
                    d.lat.values,
                    c=z,
                    cmap=cm,
                    norm=normalize,
                    alpha=0.7,
                    transform=maps.identity,
                )

                cbar1 = plt.colorbar(one)
                cbar1.set_label("Speed", rotation=270)

                ax2 = fig.add_subplot(gs[2])

                Q1 = np.quantile(d.speed_knots, 0.25)
                Q3 = np.quantile(d.speed_knots, 0.75)
                IQR = Q3 - Q1
                df_noOutliers = d.speed_knots[
                    ~(
                        (d.speed_knots < (Q1 - 1.5 * IQR))
                        | (d.speed_knots > (Q3 + 1.5 * IQR))
                    )
                ]

                ax2.hist(d.speed_knots, bins="auto")
                plt.xlabel("Speed")
                plt.ylabel("Count")

                print(ssvid, i)
                plt.show()
                print("\n")


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
df = gbq(q)

# %%
plot_track_speed(df)


# %%
def plot_double_track_speed(df):
    """
    Function to plot vessels track points colored by speed, with a histogram of speed
    """

    with pyseas.context(styles.light):

        for ssvid in df.ssvid.unique():

            fig = plt.figure(figsize=(10, 14))
            gs = gridspec.GridSpec(
                ncols=1,
                nrows=5,
                width_ratios=[
                    1,
                ],
                height_ratios=[3, 3, 1, 1, 1],
            )
            plt.style.use("seaborn-whitegrid")

            ssvid_df = df[df["ssvid"] == ssvid]

            projinfo = plot_tracks.find_projection(ssvid_df.lon, ssvid_df.lat)

            ax = maps.create_map(gs[0])
            ax.set_global()
            maps.add_land()
            maps.add_countries()
            maps.add_eezs()
            maps.add_gridlines()
            maps.add_gridlabels()

            cm = plt.cm.get_cmap("RdYlBu_r")
            z = np.array(ssvid_df.speed_knots)

            normalize = mcol.Normalize(vmin=0, vmax=10, clip=True)
            zero = ax.scatter(
                ssvid_df.lon.values,
                ssvid_df.lat.values,
                c=z,
                cmap=cm,
                norm=normalize,
                alpha=0.7,
                transform=maps.identity,
            )

            cbar0 = plt.colorbar(zero)
            cbar0.set_label("Speed", rotation=270)

            ax1 = maps.create_map(
                gs[1], projection=projinfo.projection, extent=projinfo.extent
            )
            maps.add_land()
            maps.add_countries()
            maps.add_eezs()
            maps.add_gridlines()
            maps.add_gridlabels()

            one = ax1.scatter(
                ssvid_df.lon.values,
                ssvid_df.lat.values,
                c=z,
                cmap=cm,
                norm=normalize,
                alpha=0.7,
                transform=maps.identity,
            )
            cbar1 = plt.colorbar(one)
            cbar1.set_label("Speed", rotation=270)

            one_sp = fig.add_subplot(gs[2])
            one_sp.plot(ssvid_df["timestamp"], ssvid_df["lat"])
            one_sp.set_ylabel("lat")

            two_sp = fig.add_subplot(gs[3])
            two_sp.plot(ssvid_df["timestamp"], ssvid_df["lon"])
            two_sp.set_ylabel("lon")

            ax2 = fig.add_subplot(gs[4])

            Q1 = np.quantile(ssvid_df.speed_knots, 0.25)
            Q3 = np.quantile(ssvid_df.speed_knots, 0.75)
            IQR = Q3 - Q1
            df_noOutliers = ssvid_df.speed_knots[
                ~(
                    (ssvid_df.speed_knots < (Q1 - 1.5 * IQR))
                    | (ssvid_df.speed_knots > (Q3 + 1.5 * IQR))
                )
            ]

            ax2.hist(ssvid_df.speed_knots, bins="auto")
            plt.xlabel("Speed")
            plt.ylabel("Count")

            print(ssvid)
            plt.show()
            print("\n")


# %%

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
df = gbq(q)

# %%
plot_double_track_speed(df)

# %%
mind = datetime(2019, 8, 15)
maxd = datetime(2020, 1, 15)

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
df = gbq(q)

# %%
mind = datetime(2019, 8, 15)
maxd = datetime(2020, 1, 15)

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
mind = datetime(2019, 11, 15)
maxd = datetime(2019, 12, 1)

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
df = gbq(q)

# %%
mind = datetime(2019, 8, 15)
maxd = datetime(2020, 1, 15)

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

# %%

# %%
