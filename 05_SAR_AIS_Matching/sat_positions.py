# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Calculate a Satellite's Positions
#
# Satellite positions for the Radarsat2 satellite are needed to calculat the doppler shift that causes the vessels to appear in a different location if they are moving perpendicular to the direction of the satellite's travel.

import sys
from datetime import date, datetime, timedelta

import ephem
import pandas as pd
import spacetrack.operators as op
from spacetrack import SpaceTrackClient

st = SpaceTrackClient(
    identity="", password=""
)  # replace with email and password for api key!

norad_id = sys.argv[1]  # "2017-09-03"
the_date = sys.argv[2]  # "2017-09-03"


# +
# norad_id = 32382
# the_date = "2020-04-01"

# +
date_1 = datetime.strptime(the_date, "%Y-%m-%d").date()
date_2 = (datetime.strptime(the_date, "%Y-%m-%d") + timedelta(5)).date()

decay_epoch = op.inclusive_range(date_1, date_2)
# -


def get_orb(norad_id, name):
    tles = st.tle(
        norad_cat_id=norad_id,
        orderby="epoch desc",
        limit=3,
        format="tle",
        epoch=decay_epoch,
    )
    #     print(tles)
    tle = tles.split("\n")
    line1, line2 = tle[0], tle[1]
    orb = ephem.readtle(name, line1, line2)
    return orb


# +
s1b_in_orbit = True

orb = get_orb(norad_id, "satellite")

# -


# +
thedatetimes = [
    datetime.strptime(the_date, "%Y-%m-%d") + timedelta(0, i)
    for i in range(24 * 60 * 60)
]
lats = []
lons = []
times = []
altitudes = []

for t in thedatetimes:
    # calculate for 1a
    orb.compute(t.strftime("%Y/%m/%d %H:%M:%S"))
    lon = ephem.degrees(orb.sublong) * 180 / 3.1416
    lat = ephem.degrees(orb.sublat) * 180 / 3.1416
    altitude = orb.elevation
    times.append(t)
    lats.append(lat)
    lons.append(lon)
    altitudes.append(altitude)
    # now for 1b


# -


df = pd.DataFrame(
    list(zip(times, lons, lats, altitudes)), columns=["time", "lon", "lat", "altitude"]
)

df.head()

df.to_gbq(
    "satellite_positions_v20190208.sat_noradid_{}_{}".format(
        norad_id, the_date.replace("-", "")
    ),
    project_id="world-fishing-827",
    if_exists="replace",
)

print(
    "satellite_positions_v20190208.sat_noradid_{}_{}".format(
        norad_id, the_date.replace("-", "")
    )
)
