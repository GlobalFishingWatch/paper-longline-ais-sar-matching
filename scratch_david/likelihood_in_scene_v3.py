# -*- coding: utf-8 -*-
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

import math
import os
from datetime import datetime, timedelta

import cartopy
import cartopy.crs as ccrs
import cmocean
import geopandas as gpd
# # %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# import pyseas.rasters
import pandas as pd
# +
import pyseas
# import pyseas.colors
import pyseas.cm
import pyseas.maps
import pyseas.maps.rasters
import pyseas.styles
import shapely
from cartopy import config
from matplotlib import colorbar, colors
from shapely import wkt
from shapely.geometry import Point


def gbq(q):
    return pd.read_gbq(q, project_id="world-fishing-827")


# +
q = """#StandardSql
# Interpolate AIS Postitions. This produces a table that:
#   1) Serves as in input to the table that will score AIS to SAR detection matches
#   2) Identifies the most likely location of a vessel at the time of a SAR image, and
#      whether that position likes within the scene or not
#



CREATE TEMP FUNCTION start_date() AS (DATE('2019-08-09'));
CREATE TEMP FUNCTION end_date() AS (DATE('2020-01-08'));
CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
  # Format a date as YYYYMMDD
  # e.g. DATE('2018-01-01') => '20180101'
  FORMAT_DATE('%Y%m%d',
    d) );

create temp function nautical_miles_per_degree() as (60.04);


create temp function radians(x float64) as (
  3.14159265359 * x / 180
);

create temp function degrees(x float64) as (
    x * 180 / 3.14159265359
);

create temp function deglat2km() as (
  111.699
);

create temp function meters2deg() as (
   1/(deglat2km()*1000)
);

create temp function kilometers_per_nautical_mile() as (
  1.852
);

CREATE TEMP FUNCTION weight_average_lons(lon float64, lon2 float64, timeto float64, timeto2 float64) AS
(
  # Make sure that lon < 180 and > -180, and that we average across the dateline
  # appropriately
case
when lon - lon2 > 300 then ( (lon-360)*timeto2 + lon2*timeto)/(timeto+timeto2)
when lon - lon2 < -300 then ( (lon+360)*timeto2 + lon2*timeto)/(timeto+timeto2)
else (lon*timeto2 + lon2*timeto)/(timeto+timeto2) end );

CREATE TEMP FUNCTION reasonable_lon(lon float64) AS
(case when lon > 180 then lon - 360
when lon < -180 then lon + 360
else lon end
);


CREATE TEMP FUNCTION earth_radius_km(lat float64) as
-- this is super overkill. You could just use the average
-- radius of the earth. But I wanted to see if it made a difference.
-- It matters if you want > 4 significant digits.
-- But, once I made it, I didn't feel like deleting it.
-- Equation taken from https://rechneronline.de/earth-radius/
((
select
  --  R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
  pow(
  ( pow(r1*r1 * cos(B),2) + pow(r2*r2 * sin(B),2) )
  /
  ( pow(r1 * cos(B),2) + pow(r2 * sin(B), 2) )
  ,.5)
    from
    (select
    6371.001 as r1,
    6356.752 as r2,
    Radians(lat) as B)
    limit 1
));


with
######################################
-- Data sources
######################################
--
-- Satellite locations. Get the day before and after just in case there is a
-- scene at the day boundary. This table has one value every second.
sat_positions as (select time, lon, lat, altitude
           from `satellite_positions_v20190208.sat_noradid_32382_*`
            where _table_suffix between YYYYMMDD(start_date()) and YYYYMMDD(end_date()) ),
--
-- sar detections
scene_footprints as (
SELECT
distinct
TIMESTAMP_ADD(ProductStartTime, INTERVAL
cast(timestamp_diff(ProductStopTime, ProductStopTime,
SECOND)/2 as int64) SECOND) scene_timestamp,
scene_id,
footprint
FROM (select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
union all
select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
)
# where scene_id = 'RS2_20191021_015301_0074_DVWF_HH_SCS_765981_0126_30985398'
# where scene_id = 'RS2_20190831_150553_0074_DVWF_HH_SCS_753433_9526_29818035'
 ),

--
-- raster probability. "mirror_nozeroes" means that all rows with zero values
-- have been removed, and negative i values have been removed (the raster is
-- symetrical around the i axis because a vessel is just as likely to turn left
-- as it is to turn right).
prob_raster as (select * from
 `world-fishing-827.gfw_research_precursors.point_cloud_mirror_nozeroes_v20190502`
 ),
--
-- Table we are trawing AIS messages from
ais_position_table as (
  select *, _partitiontime date from
  `world-fishing-827.gfw_research.pipe_v20201001`
  where _partitiontime between timestamp(start_date()) and timestamp(end_date())
  -- and ssvid = '255806173'
),
--
-- What is a good AIS segment? Find it here, in good_segs
good_segs as ( select seg_id from
           `world-fishing-827.gfw_research.pipe_v20201001_segs` where good_seg
),
--
--
--
##################################
# Probability raster adjustments
##################################

#########################
##
## SAR subqueries
##
#########################
--
--
-- start position of the satellite for each image
start_pos as (
select
scene_id,
timestamp_sub(scene_timestamp, INTERVAL 60 second) as start_time,
lat as start_lat,
lon as start_lon,
altitude as start_altitude
from sat_positions
join scene_footprints
on timestamp_sub(timestamp_trunc(scene_timestamp, second), INTERVAL 60 second) = time
),


-- end position of the satellite for each image
end_pos as (
select
scene_id,
timestamp_add(scene_timestamp, INTERVAL 60 second) as end_time,
lat as end_lat,
lon as end_lon,
altitude as end_altitude
from sat_positions
join scene_footprints
on timestamp_add(timestamp_trunc(scene_timestamp, second), INTERVAL 60 second) = time),


-- calcuate the direction and speed and altitude of the satellite
deltas as (
select
(end_lat - start_lat) * 111 as N_km,
(end_lon - start_lon) * 111 * cos( radians(end_lat/2 +start_lat/2) ) as E_km,
end_lat/2 +start_lat/2 as avg_lat,
start_lat,
start_lon,
end_lat,
end_lon,
start_altitude,
end_altitude,
start_time,
end_time,
scene_id
from end_pos
join
start_pos
using(scene_id)),
--
-- What direction is the satellite traveling in each scene?
sat_directions as (
select
scene_id,
degrees(ATAN2(E_Km,N_km)) sat_course, -- sat course, measured clockwise from due north, in degrees
start_lat as sat_start_lat,
start_lon as sat_start_lon,
start_altitude,
end_altitude,
timestamp_diff(end_time, start_time, second) seconds,
end_lat as sat_end_lat,
end_lon as sat_end_lon,
from deltas),
--
-- Calculate speed of satellite for each scene
-- speed of satellite varies a small ammount, so don't really need to calculate
-- for each scene. But hey -- why not calculate it directly?
sat_directions_with_speed as (
select
st_distance(st_geogpoint(sat_start_lon, sat_start_lat), st_geogpoint(sat_end_lon, sat_end_lat)) -- distance between points in meters
* (earth_radius_km(sat_end_lat) + start_altitude/1000)/ earth_radius_km(sat_end_lat) -- multiply by a factor to account for satellite altitude
/ seconds  -- make it meters per second
*1/1852 * 3600 -- * 1 nautical mile / 1852 meters   * 3600 seconds/ hour
as satellite_knots,
*
from sat_directions),
--
--
--
############################
##
## AIS subquery
##
###########################
--
-- lag and lead AIS messages
ais_messages_lagged_led as (
select
  seg_id,
  lat,
  lon,
  timestamp,
  course,
  speed_knots as speed,
  ssvid,
  distance_from_shore_m,
  LEAD(timestamp,1) OVER (PARTITION BY seg_id ORDER BY timestamp) timestamp_after,
  LAG(timestamp,1) OVER (PARTITION BY seg_id ORDER BY timestamp) timestamp_before,
  date
from
  ais_position_table
where
  abs(lat) <= 90 and abs(lon) <= 180
  and seg_id in (select seg_id from good_segs)
  and speed_knots < 50
  -- ignore the really fast vessels... most are noise
  -- this could just ignore speeds of 102.3
),
--
--
-- join on image times to get before and after
best_messages as (
select
  a.ssvid ssvid, lat, lon, speed, seg_id,
  course, timestamp,
  distance_from_shore_m,
  timestamp_diff(scene_timestamp, timestamp, SECOND) / 60.0  as delta_minutes,
  scene_id,
  scene_timestamp,
  # The following two help a later join. Basically, we want to know if this position has a matching one after or before the image
  #
  (timestamp_before is not null and abs(timestamp_diff(scene_timestamp, timestamp_before, SECOND)) / 60.0 < 12*60 ) previous_exists,
  (timestamp_after is not null and abs(timestamp_diff(scene_timestamp, timestamp_after, SECOND)) / 60.0 < 12*60 ) after_exists
from
  ais_messages_lagged_led a
join
  scene_footprints
on
  abs(timestamp_diff(scene_timestamp, timestamp, SECOND)) / 60.0  < 12*60 # less than 12 hours
  and st_distance(st_geogpoint(lon, lat), st_geogfromtext(footprint)) < 200*1852 -- within 200 nautical miles of the scene
 -- Timestamps just before or after
 -- Note that it is really tricky to consider the null value cases, and it makes things a mess later
 and(
       (timestamp <= scene_timestamp # timestamp is right before the image
       AND timestamp_after > scene_timestamp )
    or (timestamp <= scene_timestamp
       and timestamp_after is null)
    or (timestamp > scene_timestamp # timestamp is right after the image
       AND timestamp_before <= scene_timestamp )
    or (timestamp > scene_timestamp
       and timestamp_before is null)
  )
),

## Get rid of times when there are two segments from one mmsi in best messages
## In the future, this will be delt with
segs_in_scenes as (
    select scene_id, seg_id from (
        select seg_id, scene_id, row_number() over (partition by scene_id, seg_id order by min_delta_minutes) row
        from
            (select scene_id, seg_id, count(*) messages, min(abs(delta_minutes)) min_delta_minutes
            from best_messages
            group by scene_id, seg_id)
        )
    where row = 1
),

best_messages_one_seg_per_vessel as (
select * from best_messages
join segs_in_scenes
using(seg_id, scene_id)
),

####################################
# Figure out adjustment to account for the friggn' doppler shift
###################################
--
-- Get the interpolated position of each vessel
-- at the moment of the SAR image
interpolated_positions as (
SELECT
  lat + cos(radians(course)) -- x component
  *speed*delta_minutes/60 -- nautical miles (knots * minutes / (60 minutes / hour) )
  / nautical_miles_per_degree()  -- divided by the nautical miles per degree lat,
  -- which is 60 nautical miles per degree (really it is 59.9, and varies by lat)
  as lat_interpolate,
  reasonable_lon(
     lon + sin(radians(course)) -- y component
     *speed*delta_minutes/60 -- nautical miles (knots * minutes / (60 minutes / hour) )
     /(nautical_miles_per_degree()*cos(radians(lat)))) -- divided by the nautical miles per degree lon,
  -- which is 60 times the cos of the lat (really it is 59.9, and varies by lat)
  as lon_interpolate,
  1000 / greatest(1.0, ABS(delta_minutes)) as scale, # scale is the number of pixels per km
  # for the raster lookup that we will use.
  # As delta minutes gets bigger, the pixels per km gets smaller, so each pixel covers more area
  *
FROM
  best_messages_one_seg_per_vessel),


--
-- Get distance from the likely position of the vessel to the satellite,
-- and the speed of the vessel perpendicular to the satellite.
interpolated_positions_compared_to_satellite as (
select
   *,
   speed * sin(radians( course - sat_course)) as vessel_speed_perpendicular_to_sat,
   st_distance(safe.st_geogpoint(lon_interpolate, lat_interpolate), -- likely location of vessel
               ST_MAKELINE( safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat), (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) ) ) ) -- line of satellite
               / 1852 -- convert from meters to nautical miles, because
               as vessel_distance_to_sat_nm,
  cos(radians(course)) as cosdegrees,
  sin(radians(course)) as sindegrees,
from
  interpolated_positions
join
  sat_directions_with_speed
using(scene_id)
),
--
-- using satellite speed, vessel speed perpendicular to satellite direction of travel,
-- and the distance of the vessel to the satellite, calculate the distance the vessel
-- will be offset in the direction of the satellite is traveling.
interpolated_positions_adjust_formula as (
select
 *,
# 1000/scale gives the meters per pixel for the lookup table
# we are going to extrapolate by 500 pixels, so 500*1000/scale = meters of half the width
# of the raster we are using for a looup table.
(cosdegrees + sindegrees) * 500*1000/scale*meters2deg() lat11, # actual order here might be wrong...
(cosdegrees - sindegrees) * 500*1000/scale*meters2deg() lat12,

# when farther from equator, cos(lat_interpolate) will be <1, so the
# actual distance in degrees lon will be bigger, so divide by cos(lat)
(cosdegrees + sindegrees) * 500*1000/scale*meters2deg() / cos(radians(lat_interpolate))  lon11, # actual order here might be wrong...
(cosdegrees - sindegrees) * 500*1000/scale*meters2deg() / cos(radians(lat_interpolate))  lon12,
 vessel_speed_perpendicular_to_sat / satellite_knots
 * pow( ( pow(vessel_distance_to_sat_nm,2) + pow(start_altitude,2)/pow(1852,2) ) , .5)
  -- divide by 1852 to convert meters to nautical miles,
  -- then use pathangerean theorm to get the approximate distance to the satellite in
  -- nautical miles.
  as adjusted_nautical_miles_parallel_to_sat,
  # The look angle is the angle from vertical to the object, from the satellite
  degrees(atan2(vessel_distance_to_sat_nm,start_altitude/1852)) look_angle
from
  interpolated_positions_compared_to_satellite
),
--
--
-- Adjust each lat and lon by the doppler shift. Note the subtraction. If a vessel is traveling
-- perpendicular to the satellite's motion, going away from the satellite, the vessel will
-- appear offset parallel to the satellites motion opposite the direction the vessel is traveling.
-- Believe me! It works!
best_messages_adjusted as (
select
  *,
  - adjusted_nautical_miles_parallel_to_sat * cos(radians(sat_course))/60 lat_doppler_offset, -- 60 nautical miles per degree
  - adjusted_nautical_miles_parallel_to_sat * sin(radians(sat_course))/(60 * cos(radians(lat))) lon_doppler_offset, -- 60 nautical miles * cos(lat) per degree
  lat_interpolate - adjusted_nautical_miles_parallel_to_sat * cos(radians(sat_course))/60 lat_interpolate_adjusted, -- 60 nautical miles per degree
  lon_interpolate - adjusted_nautical_miles_parallel_to_sat * sin(radians(sat_course))/(60 * cos(radians(lat))) lon_interpolate_adjusted, -- 60 nautical miles * cos(lat) per degree
from
  interpolated_positions_adjust_formula
),



with_max_min_latlon as (

select
scene_id,
ssvid,
seg_id,
lat,
lon,
course,
timestamp,
scene_timestamp,
lat_doppler_offset,
lon_doppler_offset,
lat_interpolate,
lon_interpolate,
lat_interpolate_adjusted,
lon_interpolate_adjusted,
sat_course,
look_angle,
speed,
delta_minutes,
# sindegrees,
# cosdegrees,
# lat11,
# lat12,
# lon11,
# lon12,
greatest(lat11,lat12,-lat11,-lat12) + lat_interpolate_adjusted as max_lat_raster_lookup,
least(   lat11,lat12,-lat11,-lat12) + lat_interpolate_adjusted as min_lat_raster_lookup,
greatest(lon11,lon12,-lon11,-lon12) + lon_interpolate_adjusted as max_lon_raster_lookup,
least(   lon11,lon12,-lon11,-lon12) + lon_interpolate_adjusted as min_lon_raster_lookup,
previous_exists,
after_exists,
scale
from best_messages_adjusted),


before_after_joined as (
select
scene_id,
ssvid,
seg_id,
scene_timestamp,
sat_course,
a.lat lat1,
a.lon lon1,
a.course course1,
a.timestamp timestamp1,
a.lat_doppler_offset lat_doppler_offset1,
a.lon_doppler_offset lon_doppler_offset1,
a.lat_interpolate lat_interpolate1,
a.lon_interpolate lon_interpolate1,
st_geogpoint(a.lon_interpolate, a.lat_interpolate) interpolate_point1,
a.speed speed1,
a.delta_minutes delta_minutes1,
a.max_lat_raster_lookup max_lat_raster_lookup1,
a.min_lat_raster_lookup min_lat_raster_lookup1,
a.max_lon_raster_lookup max_lon_raster_lookup1,
a.min_lon_raster_lookup min_lon_raster_lookup1,
a.scale scale1,
b.lat lat2,
b.lon lon2,
b.course course2,
b.timestamp timestamp2,
b.lat_doppler_offset lat_doppler_offset2,
b.lon_doppler_offset lon_doppler_offset2,
b.lat_interpolate lat_interpolate2,
b.lon_interpolate lon_interpolate2,
st_geogpoint(b.lon_interpolate, b.lat_interpolate) interpolate_point2,
b.speed speed2,
b.delta_minutes delta_minutes2,
b.max_lat_raster_lookup max_lat_raster_lookup2,
b.min_lat_raster_lookup min_lat_raster_lookup2,
b.max_lon_raster_lookup max_lon_raster_lookup2,
b.min_lon_raster_lookup min_lon_raster_lookup2,
b.scale scale2,
(a.lat*a.delta_minutes + b.lat*abs(b.delta_minutes))/(a.delta_minutes+abs(b.delta_minutes)) lat_center,
reasonable_lon(weight_average_lons(a.lon, b.lon, a.delta_minutes,abs(b.delta_minutes))) lon_center,
a.look_angle look_angle1,
b.look_angle look_angle2
from
(select * from with_max_min_latlon where delta_minutes >=0) a
full outer join
(select * from with_max_min_latlon where delta_minutes <0) b
using(scene_id,scene_id,
ssvid,
seg_id,
scene_timestamp,
sat_course)

),

most_likely_location as

(
select
*,
# The exact logic here could be changed a bit, as it
# basically uses the interpolated position between the points
# if they the time is roughly in between the two points
case
when delta_minutes1 is null then interpolate_point2
when delta_minutes2 is null then interpolate_point1
when delta_minutes1 / (delta_minutes1 + abs(delta_minutes2)) between .25 and .75
then st_geogpoint(lon_center, lat_center)
else if(delta_minutes1 < abs(delta_minutes2), interpolate_point1, interpolate_point2 )
end as likely_location
from
before_after_joined
),
--
with_contains as (
select *,
    st_contains(footprint_geo, likely_location ) within_footprint,
    st_contains(footprint_geo, likely_location )
    and not ST_DWITHIN(ST_BOUNDARY(footprint_geo),
                         likely_location, 5000) within_footprint_5km,
    -- is it more than 1km within the scene?
    st_contains(footprint_geo, likely_location )
    and not ST_DWITHIN(ST_BOUNDARY(footprint_geo),
                         likely_location, 1000) within_footprint_1km,
 from
most_likely_location
join(select ST_GeogFromText(footprint) footprint_geo, scene_id from scene_footprints)
using(scene_id)
where
-- This looks at whether the area of the likely lookup table interesects with the scene. it is verbose,
-- but it just makes a polygon with the bounding box max and min lat and lon and sees if that intersects
-- with the scene footprint
(ST_INTERSECTS ( ST_MAKEPOLYGON(ST_MAKELINE([
                                   ST_GEOGPOINT(max_lon_raster_lookup1 + .005, max_lat_raster_lookup1 + .005),
                                   ST_GEOGPOINT(max_lon_raster_lookup1 + .005, min_lat_raster_lookup1 - .005),
                                   ST_GEOGPOINT(min_lon_raster_lookup1 - .005, min_lat_raster_lookup1 - .005),
                                   ST_GEOGPOINT(min_lon_raster_lookup1 - .005, max_lat_raster_lookup1 + .005),
                                   ST_GEOGPOINT(max_lon_raster_lookup1 + .005, max_lat_raster_lookup1 + .005)
                                   ]) ),

                footprint_geo) or max_lon_raster_lookup1 is null)

and
(ST_INTERSECTS ( ST_MAKEPOLYGON(ST_MAKELINE([
                                   ST_GEOGPOINT(max_lon_raster_lookup2 + .005, max_lat_raster_lookup2 + .005),
                                   ST_GEOGPOINT(max_lon_raster_lookup2 + .005, min_lat_raster_lookup2 - .005),
                                   ST_GEOGPOINT(min_lon_raster_lookup2 - .005, min_lat_raster_lookup2 - .005),
                                   ST_GEOGPOINT(min_lon_raster_lookup2 - .005, max_lat_raster_lookup2 + .005),
                                   ST_GEOGPOINT(max_lon_raster_lookup2 + .005, max_lat_raster_lookup2 + .005)
                                   ]) ),

                footprint_geo) or max_lon_raster_lookup2 is null)
)

## And put it together...
select
scene_id,
ssvid,
seg_id,
scene_timestamp,
st_x(likely_location) likely_lon,
st_y(likely_location) likely_lat,
lat1,
lon1,
course1,
timestamp1,
lat_doppler_offset1,
lon_doppler_offset1,
lat_interpolate1,
lon_interpolate1,
speed1,
delta_minutes1,
max_lat_raster_lookup1,
min_lat_raster_lookup1,
max_lon_raster_lookup1,
min_lon_raster_lookup1,
scale1,
lat2,
lon2,
course2,
timestamp2,
lat_doppler_offset2,
lon_doppler_offset2,
lat_interpolate2,
lon_interpolate2,
speed2,
delta_minutes2,
max_lat_raster_lookup2,
min_lat_raster_lookup2,
max_lon_raster_lookup2,
min_lon_raster_lookup2,
scale2,
lat_center,
lon_center,
within_footprint,
within_footprint_5km,
within_footprint_1km,
look_angle1,
look_angle2,
sat_course,
from
with_contains
"""


# with open('temp.sql', 'w') as f:
#     f.write(q)

# command = '''cat temp.sql | bq query --replace \
#         --destination_table=scratch_david.interp_test\
#          --allow_large_results --use_legacy_sql=false'''
# os.system(command)

# os.system("rm -f temp.sql")

# df = gbq(q)

# +
q = """
#StandardSql
# Match AIS vessel detections to Sentinel-1 vessel detections

# CREATE TEMP FUNCTION start_date() AS (DATE_SUB(DATE('{{ start_date }}'),INTERVAL 1 DAY ));
# CREATE TEMP FUNCTION end_date() AS (DATE_ADD(DATE('{{ end_date }}'), INTERVAL 1 DAY));
CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
  # Format a date as YYYYMMDD
  # e.g. DATE('2018-01-01') => '20180101'
  FORMAT_DATE('%Y%m%d',
    d) );

CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
y1 float64, y2 float64, x float64, y float64) AS (
    # see https://en.wikipedia.org/wiki/Bilinear_interpolation
    # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
    # x and y are the coordinates we want the value for
    1/( (x2 - x1) * (y2 - y1)) *
      (
         Q11 * (x2 - x) * (y2 - y)
       + Q21 * (x - x1) * (y2 - y)
       + Q12 * (x2 - x) * (y - y1)
       + Q22 * (x - x1) * (y - y1)
      )
    );

create temp function colmax(x float64, y float64) as (
  if(x > y, x, y)
);

create temp function radians(x float64) as (
  3.14159265359 * x / 180
);

create temp function deglat2km() as (
  111.699
);


create temp function kilometers_per_nautical_mile() as (
  1.852
);

create temp function map_label(label string)
as (
  case when label ="drifting_longlines" then "drifting_longlines"
  when label ="purse_seines" then "purse_seines"
  when label ="other_purse_seines" then "purse_seines"
  when label ="tuna_purse_seines" then "purse_seines"
  when label ="cargo_or_tanker" then "cargo_or_tanker"
  when label ="cargo" then "cargo_or_tanker"
  when label ="tanker" then "cargo_or_tanker"
  when label ="tug" then "tug"
  when label = "trawlers" then "trawlers"
  else "other" end
);

create temp function map_speed(x float64) as (
  case
  when x < 2 then 0
  when x < 4 then 2
  when x < 6 then 4
  when x < 8 then 6
  when x < 10 then 8
  else 10 end
);

create temp function map_minutes(x float64) as (
  case when x < -384 then  -512
  when x < -256 then -384
  when x < -192 then -256
  when x < -128 then -192
  when x < -96 then -128
  when x < -64 then -96
  when x < -48 then -64
  when x < -32 then -48
  when x < -24 then -32
  when x < -16 then -24
  when x < -12 then -16
  when x < -8 then -12
  when x < -6 then -8
  when x < -4 then -6
  when x < -3 then -4
  when x < -2 then -3
  when x < -1 then -2
  when x < 0 then -1
  when x < 1 then 0
  when x < 2 then 1
  when x < 3 then 2
  when x < 4 then 3
  when x < 6 then 4
  when x < 8 then 6
  when x < 12 then 8
  when x < 16 then 12
  when x < 24 then 16
  when x < 32 then 24
  when x < 48 then 32
  when x < 64 then 48
  when x < 96 then 64
  when x < 128 then 96
  when x < 192 then 128
  when x < 256 then 192
  when x < 384 then 256
  else 384 end);

CREATE TEMP FUNCTION reasonable_lon(lon float64) AS
  (case when lon > 180 then lon - 360
  when lon < -180 then lon + 360
  else lon end
);


CREATE TEMP FUNCTION earth_radius_km(lat float64) as
-- This is super overkill. You could just use the average
-- radius of the earth. But I wanted to see if it made a difference.
-- It matters if you want > 4 significant digits.
-- But, once I made it, I didn't feel like deleting it.
-- Equation taken from https://rechneronline.de/earth-radius/
((
select
  --  R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
  pow(
  ( pow(r1*r1 * cos(B),2) + pow(r2*r2 * sin(B),2) )
  /
  ( pow(r1 * cos(B),2) + pow(r2 * sin(B), 2) )
  ,.5)
    from
    (select
    6371.001 as r1,
    6356.752 as r2,
    Radians(lat) as B)
    limit 1
));


with
######################################
-- Data sources
######################################
--
-- vessel_interpolated_positions
-- This is a table created through a previous query that gets the interpolated position
-- of vessels that could be within a given scene. It also includes offset for doppler effect
-- that accounts for satellite direction and speed relative to the vessel

vessel_interpolated_positions as
 (select * from scratch_david.interp_test),


--
-- sar detections
sar_detections as (
SELECT TIMESTAMP_ADD(ProductStartTime, INTERVAL cast(timestamp_diff(ProductStopTime, ProductStopTime, SECOND)/2 as int64) SECOND) detect_timestamp,
 DetectionId as detect_id,
 lon detect_lon,
 lat detect_lat,
 scene_id
 FROM
  proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 SELECT TIMESTAMP_ADD(ProductStartTime, INTERVAL cast(timestamp_diff(ProductStopTime, ProductStopTime, SECOND)/2 as int64) SECOND) detect_timestamp,
 DetectionId as detect_id,
 lon detect_lon,
 lat detect_lat,
 scene_id
 from
 proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
 ),

--
-- raster probability. "mirror_nozeroes" means that all rows with zero values
-- have been removed, and negative i values have been removed (the raster is
-- symetrical around the i axis because a vessel is just as likely to turn left
-- as it is to turn right).
prob_raster as (select * from
 `world-fishing-827.gfw_research_precursors.point_cloud_mirror_nozeroes_v20190502`
 ),
--
--
-- vessel info table, used for vessel class.
-- acceptable vessel classes are "trawlers","purse_seines","tug","cargo_or_tanker","drifting_longlines","fishing", and "other"
vessel_info_table as (select
        ssvid,
        case when best.best_vessel_class in ("trawlers",
                                            "purse_seines",
                                             "tug","cargo_or_tanker","cargo" , "tanker",
                                             "drifting_longlines") then best.best_vessel_class
        when on_fishing_list_best then "fishing"
        else "other"
        end label,
        from `world-fishing-827.gfw_research.vi_ssvid_v20200410` ),
--
--
##################################
# Probability raster adjustments
##################################
--
-- get the average for general fishing. This is creating a composite category
-- of other categories.
probability_table as (select
   probability,labels,minutes_lower,speed_lower,i,j
from
  prob_raster
union all
select
   avg(probability) probability,"fishing" labels, minutes_lower, speed_lower,i,j
from
  prob_raster
where labels in ("trawlers","purse_seines","drifting_longlines")
group by
   labels,minutes_lower,speed_lower,i,j),
--
--
-- weight is a measure of how concetrated the raster is, and is used to assess score
weights as (
select
  # this is because the value at 0 is not mirrored
  sum(if(i>0, 2*probability * probability, probability * probability )) as weight,
  labels,
  minutes_lower,
  speed_lower
from
  probability_table
group by
  labels,
  minutes_lower,
  speed_lower
),
--
-- combine probabilities and weights into one table
probabilities_and_weights as (
select * from
  probability_table
join
  weights
using(labels, minutes_lower,speed_lower)
),
--
--
-- the raster has only positive i values because it is symmetrical
-- around the i axis (the boat is as equally likely to turn left as right).
-- Make a full raster with negative values for the later join.
probabilities_and_weights_neg as (
select
labels, minutes_lower, speed_lower, probability, i, j, weight
from
probabilities_and_weights
union all
select
-- same except take negative i!
labels, minutes_lower, speed_lower, probability, -i as i, j, weight
from
probabilities_and_weights
where i >0
),

######
#
################


interpolated_unstacked as (
select
scene_id,
scene_timestamp,
ssvid,
timestamp1 as timestamp,
lat1 + lat_doppler_offset1 as lat,
lon1 + lon_doppler_offset1 as lon,
delta_minutes1 as delta_minutes,
speed1 as speed,
scale1 as scale,
course1 as course,
min_lon_raster_lookup1 as min_lon,
min_lat_raster_lookup1 as min_lat,
max_lon_raster_lookup1 as max_lon,
max_lat_raster_lookup1 as max_lat,
lat_interpolate2 is null as is_single
from vessel_interpolated_positions
where lat_interpolate1 is not null
union all
select
scene_id,
scene_timestamp,
ssvid,
timestamp2 as timestamp,
lat2 + lat_doppler_offset2 as lat,
lon2 + lon_doppler_offset2 as lon,
delta_minutes2 as delta_minutes,
speed2 as speed,
scale2 as scale,
course2 as course,
min_lon_raster_lookup2 as min_lon,
min_lat_raster_lookup2 as min_lat,
max_lon_raster_lookup2 as max_lon,
max_lat_raster_lookup2 as max_lat,
lat_interpolate1 is null as is_single
from vessel_interpolated_positions
where lat_interpolate2 is not null
),

# Add vessel class
interpolated_unstacked_w_label as (
    select * from interpolated_unstacked
    join vessel_info_table using(ssvid)
),


detects_vs_interpolated as (
select
ssvid,
label,
scene_id,
detect_id,
detect_lat,
detect_lon,
lat,
lon,
timestamp,
delta_minutes,
speed,
scale,
course,
is_single
from interpolated_unstacked_w_label
join
 sar_detections
 using(scene_id)
where detect_lat between min_lat and max_lat
and detect_lon between min_lon and max_lon
),

--
####################
# joins to the probability raster
###################
--
key_query_1 as (
select *,
  deglat2km() * (detect_lon - lon) * cos(radians(lat)) as u,
  deglat2km() * (detect_lat - lat) as v,
  radians(course) as course_rads
from
  detects_vs_interpolated
),
--
-- rotate the coordinates
key_query_2 as (
select
  *,
  cos(course_rads) * u - sin(course_rads) * v as x,
  cos(course_rads) * v + sin(course_rads) * u as y
  -- rotation of coordinates, described here: https://en.wikipedia.org/wiki/Rotation_of_axes
  -- Note that our u and v / x and y are switched from the standard way to measure
  -- this, largely because vessels measure course from due north, moving clockwise,
  -- while mosth math measures angle from the x axis counterclockwise. Annoying!
  --
#   1000 / colmax(1.0, ABS(delta_minutes)) as scale
#     This is the python function we are copying here:
#      def scale(dt):
#         return 1000.0 / max(1, abs(dt))
from
  key_query_1
),
--
-- adjust by scale -- that is the probability raster will be scalled
-- based on how long before or after the ping the image was taken.
-- Also, move the raster so that 0,0 is where the vessel would be
-- if it traveled in a straight line.
key_query_3 as
(
select * from (
  select *,
    x * scale as x_key,
    (y - speed*kilometers_per_nautical_mile()*delta_minutes/60 ) * scale  as y_key, -- the old version interpolated, and this one doesn't
    # y * scale as y_key,
    # Map these values to the values in the probability rasters
    map_speed(speed) as speed_key,
    map_minutes(delta_minutes) as minute_key,
    map_label(label) as label_key
  from
    key_query_2
  )
where abs(x_key) <=500 and abs(y_key) <=500
),
--
--
-- Match to probability, and interpolate between
-- the four closest values. This bilinear interpoloation
-- in theory allows us to reduce the size of the raster we are joining on
messages_with_probabilities as
(
select
  -- this would get the value exact, the weight_scaled
  -- / pow((1000/(colmax( 1, probs.minutes_lower/2 + probs.minutes_upper /2))),2) * scale*scale
  * except(i, j, probability),
  bilinear_interpolation(
  ifnull(probs_11.probability,0),
  ifnull(probs_12.probability,0),
  ifnull(probs_22.probability,0),
  ifnull(probs_21.probability,0),
  cast(x_key - .5 as int64), cast(x_key + .5 as int64),
  cast(y_key - .5 as int64), cast(y_key + .5 as int64) ,
  x_key, y_key) as probability,
  -- to get at least one value.
  -- weight *should* be the same for each, but we need to make sure it isn't null
  case when probs_11.weight is not null then probs_11.weight/(scale*scale)
  when probs_12.weight is not null then probs_12.weight/(scale*scale)
  when probs_22.weight is not null then probs_22.weight/(scale*scale)
  when probs_21.weight is not null then probs_21.weight/(scale*scale)
  else 0
  end
  as weight_scaled
from
  key_query_3
left join
  --
  -- joining on four locaitons to do bilinear interpolation
probabilities_and_weights_neg  as probs_11
  on  probs_11.i = cast(x_key - .5 as int64) and probs_11.j = cast(y_key - .5 as int64)
  and probs_11.speed_lower = cast(speed_key as int64)
  and probs_11.minutes_lower = cast(minute_key as int64)
  and probs_11.labels = label_key
left join
probabilities_and_weights_neg  as probs_12
  on  probs_12.i = cast(x_key -.5 as int64) and probs_12.j = cast(y_key + .5 as int64)
  and probs_12.speed_lower = cast(speed_key as int64)
  and probs_12.minutes_lower = cast(minute_key as int64)
  and probs_12.labels = label_key
left join
probabilities_and_weights_neg  as probs_22
  on  probs_22.i = cast(x_key +.5 as int64) and probs_22.j = cast(y_key + .5 as int64)
  and probs_22.speed_lower = cast(speed_key as int64)
  and probs_22.minutes_lower = cast(minute_key as int64)
  and probs_22.labels = label_key
left join
probabilities_and_weights_neg  as probs_21
  on  probs_21.i = cast(x_key +.5 as int64) and probs_21.j = cast(y_key - .5 as int64)
  and probs_21.speed_lower = cast(speed_key as int64)
  and probs_21.minutes_lower = cast(minute_key as int64)
  and probs_21.labels = label_key
),

joined_back_with_interpolated_positions as (
  select
  ssvid,
  scene_id,
  ifnull(probability,0) probability,
  ifnull(weight_scaled, 0) weight_scaled,
  a.scale,
  detect_id,
  detect_lon,
  detect_lat,
  scene_timestamp as detect_timestamp,
  a.timestamp,
  a.course,
  a.lat,
  a.lon,
  a.speed,
  a.is_single,
  delta_minutes
  from
  interpolated_unstacked a
  left join
  messages_with_probabilities b
  using(ssvid, scene_id, delta_minutes)
),
--
--
--
joined_detects AS (
SELECT
  a.ssvid,
  a.scene_id scene_id,
  a.detect_id detect_id,
  ifnull(a.detect_timestamp, b.detect_timestamp) detect_timestamp,
  if(a.probability is null, 0, a.probability ) a_probability,
  if(b.probability is null, 0, b.probability ) b_probability,
  if(a.weight_scaled is null, 0, a.weight_scaled ) a_weight_scaled,
  if(b.weight_scaled is null, 0, b.weight_scaled ) b_weight_scaled,
  a.scale a_scale,
  b.scale b_scale,
  a.detect_lat detect_lat,
  a.detect_lon detect_lon,
  a.timestamp a_timestamp,
  b.timestamp b_timestamp,
  a.speed a_speed,
  b.speed b_speed,
  a.course a_course,
  b.course b_course,
  a.lat a_lat,
  a.lon a_lon,
  b.lat b_lat,
  b.lon b_lon,
  ifnull(a.is_single, b.is_single) is_single

FROM
  (select * from  joined_back_with_interpolated_positions where delta_minutes >= 0) a
full outer join
  (select * from joined_back_with_interpolated_positions where delta_minutes < 0) b
using(ssvid, detect_id)
),
--
-- Apply a score to each detection to vessel
-- This score was figured out by Tim Hochberg, who tried
-- out a series of different things.
 scored_detects as (
select *,
   safe_divide( (a_probability*a_weight_scaled +
    b_probability*b_weight_scaled),
a_weight_scaled + b_weight_scaled) AS score
from joined_detects)
# --
# --
# #################################################
# # And the answer is....
# #################################################
# --

select "ais" as source, * from scored_detects where score > 0

"""

with open("temp.sql", "w") as f:
    f.write(q)

command = """cat temp.sql | bq query --replace \
        --destination_table=scratch_david.score_test\
         --allow_large_results --use_legacy_sql=false"""
os.system(command)

os.system("rm -f temp.sql")


# +
scored_table = "scratch_david.score_test"

q = f"""
with scored_table as (select concat(ssvid, source) as ssvid_source, * from {scored_table})

SELECT
  *,
  ROW_NUMBER() OVER (PARTITION BY detect_id ORDER BY score DESC) row_number_detect_id,
  ROW_NUMBER() OVER (PARTITION BY ssvid_source, scene_id ORDER BY score DESC) row_number_ssvid
FROM
  scored_table
  """


with open("temp.sql", "w") as f:
    f.write(q)

command = """cat temp.sql | bq query --replace \
        --destination_table=scratch_david.ranked_test\
         --allow_large_results --use_legacy_sql=false"""
os.system(command)

os.system("rm -f temp.sql")


# +
q = """

with

scores_ranked as (select * from

-- scratch_david.m_mda_equador_2ranked where date(_partitiontime) = "2020-04-01"
scratch_david.ranked_test

),


objects_table as
(
SELECT TIMESTAMP_ADD(ProductStartTime, INTERVAL cast(timestamp_diff(ProductStopTime, ProductStopTime, SECOND)/2 as int64) SECOND) detect_timestamp,
 DetectionId as detect_id,
 lon detect_lon,
 lat detect_lat,
 scene_id
 FROM
  proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 SELECT TIMESTAMP_ADD(ProductStartTime, INTERVAL cast(timestamp_diff(ProductStopTime, ProductStopTime, SECOND)/2 as int64) SECOND) detect_timestamp,
 DetectionId as detect_id,
 lon detect_lon,
 lat detect_lat,
 scene_id
 from
 proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110

 ),



# join on shipping lanes
detections_and_shipping_lanes as (
select
detect_lat,
detect_lon,
scene_id,
detect_timestamp,
detect_id,
-- ifnull(e.hours,0) AS cargo_tanker_hours_in_2018_per_50th_of_degree,
-- ifnull(g.hours,0) AS cargo_tanker_hours_in_2018_per_20th_of_degree,
 ifnull(h.hours,0) AS cargo_tanker_hours_in_2018_per_10th_of_degree
from
objects_table
 left JOIN
   `gfw_research_precursors.10km_cargotanker_2018_density_v20190523` h
 ON
   CAST(FLOOR(detect_lat*10) AS int64) = h.lat_bin
   AND CAST(FLOOR(detect_lon*10) AS int64) = h.lon_bin
      )
      ,
#
#




   top_matches as (
  select * from scores_ranked
  where row_number_detect_id = 1 and row_number_ssvid = 1),

  second_matches_ranked as (
     select *, row_number() over
    (partition by detect_id order by score desc) row_number_detect_id_2nd,
     row_number() over
    (partition by ssvid,scene_id order by score desc) row_number_ssvid_2nd
  from scores_ranked
  where concat(ssvid,scene_id) not in (select concat(ssvid,scene_id) from top_matches)
  and detect_id not in (select detect_id from top_matches)),

  second_matches as (
  select * from second_matches_ranked where row_number_detect_id_2nd = 1 and row_number_ssvid_2nd = 1),

  third_matches_ranked as
  (
     select *, row_number() over
    (partition by detect_id order by score desc) row_number_detect_id_3rd,
     row_number() over
    (partition by ssvid, scene_id order by score desc) row_number_ssvid_3rd
  from second_matches_ranked
  where
  concat(ssvid,scene_id) not in (select concat(ssvid,scene_id)  from second_matches)
  and detect_id not in (select detect_id from second_matches)
  ),

  third_matches as (
 select * from third_matches_ranked where row_number_detect_id_3rd = 1 and row_number_ssvid_3rd = 1),

 forth_matches_ranked as   (
     select *, row_number() over
    (partition by detect_id order by score desc) row_number_detect_id_4th,
     row_number() over
    (partition by ssvid,scene_id order by score desc) row_number_ssvid_4th
  from third_matches_ranked
  where
  concat(ssvid,scene_id) not in (select concat(ssvid,scene_id) from third_matches)
  and detect_id not in (select detect_id from third_matches)
  ),

  fourth_matches as (
  select * from forth_matches_ranked where row_number_detect_id_4th = 1 and row_number_ssvid_4th = 1),

  top_4_matches as (
  select * from fourth_matches
  union all
  select *, null as row_number_detect_id_4th ,
  null as row_number_ssvid_4th from third_matches
  union all
  select *,
  null as row_number_detect_id_4th ,
  null as row_number_ssvid_4th,
  null as row_number_detect_id_3rd ,
  null as row_number_ssvid_3rd
  from second_matches
  union all
  select *,
  null as row_number_detect_id_4th ,
  null as row_number_ssvid_4th,
  null as row_number_detect_id_3rd ,
  null as row_number_ssvid_3rd ,
  null as row_number_detect_id_2nd ,
  null as row_number_ssvid_2nd
  from top_matches
  order by
   row_number_detect_id,
   row_number_ssvid)


   select
   b.scene_id scene_id,
   b.detect_lon detect_lon,
   b.detect_lat detect_lat,
   b.detect_timestamp detect_timestamp,
   * except(scene_id, score, detect_lon, detect_lat, detect_timestamp, ssvid_source,
  row_number_detect_id,row_number_ssvid,row_number_detect_id_2nd,row_number_ssvid_2nd,
  row_number_detect_id_3rd,row_number_ssvid_3rd,row_number_detect_id_4th,row_number_ssvid_4th),
   ifnull(score,0) score
   from top_4_matches a
   full outer join detections_and_shipping_lanes b
   using(detect_id)

"""


with open("temp.sql", "w") as f:
    f.write(q)

command = """cat temp.sql | bq query --replace \
        --destination_table=scratch_david.top_test\
         --allow_large_results --use_legacy_sql=false"""
os.system(command)

os.system("rm -f temp.sql")
# -


# +
q = """#StandardSql

CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
y1 float64, y2 float64, x float64, y float64) AS (
    # see https://en.wikipedia.org/wiki/Bilinear_interpolation
    # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
    # x and y are the coordinates we want the value for
    1/( (x2 - x1) * (y2 - y1)) *
      (
         Q11 * (x2 - x) * (y2 - y)
       + Q21 * (x - x1) * (y2 - y)
       + Q12 * (x2 - x) * (y - y1)
       + Q22 * (x - x1) * (y - y1)
      )
    );


create temp function radians(x float64) as (
  3.14159265359 * x / 180
);

create temp function deglat2km() as (
  111.699
);

create temp function meters2deg() as (
   1/(deglat2km()*1000)
);

create temp function kilometers_per_nautical_mile() as (
  1.852
);

create temp function one_over_cellsize() as (
  200  -- 500 meter resolution roughly
);

create temp function map_label(label string)
as (
  case when label ="drifting_longlines" then "drifting_longlines"
  when label ="purse_seines" then "purse_seines"
  when label ="other_purse_seines" then "purse_seines"
  when label ="tuna_purse_seines" then "purse_seines"
  when label ="cargo_or_tanker" then "cargo_or_tanker"
  when label ="cargo" then "cargo_or_tanker"
  when label ="tanker" then "cargo_or_tanker"
  when label ="tug" then "tug"
  when label = "trawlers" then "trawlers"
  else "other" end
);

create temp function map_speed(x float64) as (
  case
  when x < 2 then 0
  when x < 4 then 2
  when x < 6 then 4
  when x < 8 then 6
  when x < 10 then 8
  else 10 end
);

create temp function map_minutes(x float64) as (
  case when x < -384 then  -512
  when x < -256 then -384
  when x < -192 then -256
  when x < -128 then -192
  when x < -96 then -128
  when x < -64 then -96
  when x < -48 then -64
  when x < -32 then -48
  when x < -24 then -32
  when x < -16 then -24
  when x < -12 then -16
  when x < -8 then -12
  when x < -6 then -8
  when x < -4 then -6
  when x < -3 then -4
  when x < -2 then -3
  when x < -1 then -2
  when x < 0 then -1
  when x < 1 then 0
  when x < 2 then 1
  when x < 3 then 2
  when x < 4 then 3
  when x < 6 then 4
  when x < 8 then 6
  when x < 12 then 8
  when x < 16 then 12
  when x < 24 then 16
  when x < 32 then 24
  when x < 48 then 32
  when x < 64 then 48
  when x < 96 then 64
  when x < 128 then 96
  when x < 192 then 128
  when x < 256 then 192
  when x < 384 then 256
  else 384 end);

CREATE TEMP FUNCTION reasonable_lon(lon float64) AS
  (case when lon > 180 then lon - 360
  when lon < -180 then lon + 360
  else lon end
);


CREATE TEMP FUNCTION earth_radius_km(lat float64) as
-- this is super overkill. You could just use the average
-- radius of the earth. But I wanted to see if it made a difference.
-- It matters if you want > 4 significant digits.
-- But, once I made it, I didn't feel like deleting it.
-- Equation taken from https://rechneronline.de/earth-radius/
((
select
  --  R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
  pow(
  ( pow(r1*r1 * cos(B),2) + pow(r2*r2 * sin(B),2) )
  /
  ( pow(r1 * cos(B),2) + pow(r2 * sin(B), 2) )
  ,.5)
    from
    (select
    6371.001 as r1,
    6356.752 as r2,
    Radians(lat) as B)
    limit 1
));


with
######################################
-- Data sources
######################################
--
-- interpolated positions
-- vessel info table, used for vessel class.
-- acceptable vessel classes are "trawlers","purse_seines","tug","cargo_or_tanker","drifting_longlines","fishing", and "other"
vessel_info_table as (select
        ssvid,
        case when best.best_vessel_class in ("trawlers",
                                            "purse_seines",
                                             "tug","cargo_or_tanker",
                                             "tanker","cargo",
                                             "drifting_longlines") then best.best_vessel_class
        when on_fishing_list_best then "fishing"
        -- when best.best_vessel_class = "gear" then "gear"
        else "other"
        end label,
        best.best_vessel_class = 'gear' is_gear
        from `world-fishing-827.gfw_research.vi_ssvid_v20200410` ),

interp_positions as (select * except(label),
                         ifnull(label, "other") label
                    from scratch_david.interp_test
                      left join vessel_info_table
                      using(ssvid)),


-- sar detections
scene_footprints as (
SELECT
distinct
TIMESTAMP_ADD(ProductStartTime, INTERVAL
cast(timestamp_diff(ProductStopTime, ProductStopTime,
SECOND)/2 as int64) SECOND) scene_timestamp,
scene_id,
footprint
FROM (select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
union all
select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
)
 ),

--
-- raster probability. "mirror_nozeroes" means that all rows with zero values
-- have been removed, and negative i values have been removed (the raster is
-- symetrical around the i axis because a vessel is just as likely to turn left
-- as it is to turn right).
prob_raster as (select * from
 `world-fishing-827.gfw_research_precursors.point_cloud_mirror_nozeroes_v20190502`
 ),
--

--

--
--
##################################
# Probability raster adjustments
##################################
--
-- get the average for general fishing. This is creating a composite category
-- of other categories.
probability_table as (select
   probability,labels,minutes_lower,speed_lower,i,j
from
  prob_raster
union all
select
   avg(probability) probability,"fishing" labels, minutes_lower, speed_lower,i,j
from
  prob_raster
where labels in ("trawlers","purse_seines","drifting_longlines")
group by
   labels,minutes_lower,speed_lower,i,j),
--
--

--
--
-- the raster has only positive i values because it is symmetrical
-- around the i axis (the boat is as equally likely to turn left as right).
-- Make a full raster with negative values for the later join.
probabilities_and_weights_neg as (
select
labels, minutes_lower, speed_lower, probability, i, j --, weight
from
probability_table
union all
select
-- same except take negative i!
labels, minutes_lower, speed_lower, probability, -i as i, j -- , weight
from
probability_table
where i >0
),
#########################
##
## SAR subqueries
##
#########################
--
--
just_bounds as
(select
ssvid,
label,
scene_id,
speed1 as speed,
course1 as course,
scale1 as scale,
delta_minutes2 is null as is_single,
delta_minutes1 as delta_minutes,
min_lon_raster_lookup1 as min_lon,
max_lon_raster_lookup1 as max_lon,
min_lat_raster_lookup1 as min_lat,
max_lat_raster_lookup1 as max_lat,
lat_interpolate1 + lat_doppler_offset1 as lat_interpolate_adjusted,
lon_interpolate1 + lon_doppler_offset1 as lon_interpolate_adjusted
from interp_positions where delta_minutes1 is not null
union all
select
ssvid,
label,
scene_id,
speed2 as speed,
course2 as course,
scale2 as scale,
delta_minutes1 is null as is_single,
delta_minutes2 as delta_minutes,
min_lon_raster_lookup2 as min_lon,
max_lon_raster_lookup2 as max_lon,
min_lat_raster_lookup2 as min_lat,
max_lat_raster_lookup2 as max_lat,
lat_interpolate2 + lat_doppler_offset2 as lat_interpolate_adjusted,
lon_interpolate2 + lon_doppler_offset2 as lon_interpolate_adjusted
from interp_positions where delta_minutes2 is not null
),

 lat_array AS(
  SELECT
    * ,
    lat_bin/one_over_cellsize() as detect_lat  -- to get the middle of the cell
  FROM
    just_bounds
  CROSS JOIN
    UNNEST(GENERATE_ARRAY(
           cast(FLOOR(min_lat*one_over_cellsize()) as int64) - 2, -- just to pad it a bit
           cast(FLOOR(max_lat*one_over_cellsize()) as int64) +2,
           1))
     AS lat_bin),


  lon_array AS (
  SELECT
    *,
    lon_bin/one_over_cellsize() as detect_lon -- to get the middle of the cell
  FROM
    just_bounds
  CROSS JOIN
        UNNEST(GENERATE_ARRAY(
           cast(FLOOR(min_lon*one_over_cellsize()) as int64) - 2, -- just to pad it a bit
           cast(FLOOR(max_lon*one_over_cellsize()) as int64) + 2,
           1))  AS lon_bin),



  id_lat_lon_array AS (
  select
  a.ssvid,
  a.label,
  a.scene_id,
  a.speed,
  a.course,
  a.scale,
  a.is_single,
  a.delta_minutes,
  a.lat_interpolate_adjusted,
  a.lon_interpolate_adjusted,
  lon_bin,
  lat_bin,
  detect_lon,
  detect_lat,
  FROM
    lon_array a
  CROSS JOIN
    lat_array b
  WHERE
    a.scene_id=b.scene_id
    and a.ssvid=b.ssvid
    and a.delta_minutes = b.delta_minutes),

--
####################
# joins to the probability raster
###################
--

key_query_1 as (
select *,
  deglat2km() * (detect_lon - lon_interpolate_adjusted) * cos(radians(lat_interpolate_adjusted)) as u,
  deglat2km() * (detect_lat - lat_interpolate_adjusted) as v,
  radians(course) as course_rads
from
  id_lat_lon_array
),

# --
# -- rotate the coordinates
key_query_2 as (
select
  *,
  cos(course_rads) * u - sin(course_rads) * v as x,
  cos(course_rads) * v + sin(course_rads) * u as y,
  -- rotation of coordinates, described here: https://en.wikipedia.org/wiki/Rotation_of_axes
  -- Note that our u and v / x and y are switched from the standard way to measure
  -- this, largely because vessels measure course from due north, moving clockwise,
  -- while mosth math measures angle from the x axis counterclockwise. Annoying!
  --
  # 1000 / colmax(1.0, ABS(delta_minutes)) as scale
#     This is the python function we are copying here:
#      def scale(dt):
#         return 1000.0 / max(1, abs(dt))
from
  key_query_1
),
# --
# -- adjust by scale -- that is the probability raster will be scalled
# -- based on how long before or after the ping the image was taken.
# -- Also, move the raster so that 0,0 is where the vessel would be
# -- if it traveled in a straight line.
key_query_3 as
(
select * from (
  select *,
    x * scale as x_key,
    # (y - speed*kilometers_per_nautical_mile()*delta_minutes/60 ) * scale  as y_key,
    y  * scale  as y_key, # because using interpolated position, already centered at this location
    # Map these values to the values in the probability rasters
    map_speed(speed) as speed_key,
    map_minutes(delta_minutes) as minute_key,
    map_label(label) as label_key
  from
    key_query_2
  )
where abs(x_key) <=500 and abs(y_key) <=500
),



# # --
# # --
-- Match to probability, and interpolate between
-- the four closest values. This bilinear interpoloation
-- in theory allows us to reduce the size of the raster we are joining on
messages_with_probabilities as
 (
 select
--   -- this would get the value exact, the weight_scaled
--   -- / pow((1000/(colmax( 1, probs.minutes_lower/2 + probs.minutes_upper /2))),2) * scale*scale
   * except(i, j, probability),
   bilinear_interpolation(
   ifnull(probs_11.probability,0),
   ifnull(probs_12.probability,0),
   ifnull(probs_22.probability,0),
   ifnull(probs_21.probability,0),
   cast(x_key - .5 as int64), cast(x_key + .5 as int64),
   cast(y_key - .5 as int64), cast(y_key + .5 as int64) ,
   x_key, y_key) as probability,
--   -- to get at least one value.
--   -- weight *should* be the same for each, but we need to make sure it isn't null
 from
   key_query_3
 left join
-- joining on four locaitons to do bilinear interpolation
 probabilities_and_weights_neg  as probs_11
   on  probs_11.i = cast(x_key - .5 as int64) and probs_11.j = cast(y_key - .5 as int64)
   and probs_11.speed_lower = cast(speed_key as int64)
   and probs_11.minutes_lower = cast(minute_key as int64)
   and probs_11.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_12
   on  probs_12.i = cast(x_key -.5 as int64) and probs_12.j = cast(y_key + .5 as int64)
   and probs_12.speed_lower = cast(speed_key as int64)
   and probs_12.minutes_lower = cast(minute_key as int64)
   and probs_12.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_22
   on  probs_22.i = cast(x_key +.5 as int64) and probs_22.j = cast(y_key + .5 as int64)
   and probs_22.speed_lower = cast(speed_key as int64)
   and probs_22.minutes_lower = cast(minute_key as int64)
   and probs_22.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_21
   on  probs_21.i = cast(x_key +.5 as int64) and probs_21.j = cast(y_key - .5 as int64)
   and probs_21.speed_lower = cast(speed_key as int64)
   and probs_21.minutes_lower = cast(minute_key as int64)
   and probs_21.labels = label_key
 ),

prob_norm1 as (
select *, probability/prob_sum as prob_norm from messages_with_probabilities
join
(select sum(probability) prob_sum, ssvid, scene_id, delta_minutes
from messages_with_probabilities group by ssvid, scene_id, delta_minutes
)
using(ssvid, scene_id, delta_minutes)
)

select
  ssvid,
  label,
  scene_id,
  speed,
  course,
  scale,
  is_single,
  delta_minutes,
  lat_interpolate_adjusted,
  lon_interpolate_adjusted,
  lon_bin,
  lat_bin,
  detect_lon,
  detect_lat
from prob_norm1 """

with open("temp.sql", "w") as f:
    f.write(q)

command = """cat temp.sql | bq query --replace \
        --destination_table=scratch_david.interp_raster_test\
         --allow_large_results --use_legacy_sql=false"""
os.system(command)

os.system("rm -f temp.sql")
# -

q = """# select * from scratch_david.interp_test where within_footprint_5km is true and within_footprint


#StandardSql
# Match AIS vessel detections to Sentinel-1 vessel detections

# CREATE TEMP FUNCTION start_date() AS (DATE_SUB(DATE('{{ start_date }}'),INTERVAL 1 DAY ));
# CREATE TEMP FUNCTION end_date() AS (DATE_ADD(DATE('{{ end_date }}'), INTERVAL 1 DAY));
CREATE TEMP FUNCTION YYYYMMDD(d DATE) AS (
  # Format a date as YYYYMMDD
  # e.g. DATE('2018-01-01') => '20180101'
  FORMAT_DATE('%Y%m%d',
    d) );

# create temp function scene_id() as ('');
create temp function ssvid() as ('441825000');

CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
y1 float64, y2 float64, x float64, y float64) AS (
    # see https://en.wikipedia.org/wiki/Bilinear_interpolation
    # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
    # x and y are the coordinates we want the value for
    1/( (x2 - x1) * (y2 - y1)) *
      (
         Q11 * (x2 - x) * (y2 - y)
       + Q21 * (x - x1) * (y2 - y)
       + Q12 * (x2 - x) * (y - y1)
       + Q22 * (x - x1) * (y - y1)
      )
    );

create temp function colmax(x float64, y float64) as (
  if(x > y, x, y)
);

create temp function radians(x float64) as (
  3.14159265359 * x / 180
);

create temp function deglat2km() as (
  111.699
);


create temp function kilometers_per_nautical_mile() as (
  1.852
);

create temp function map_label(label string)
as (
  case when label ="drifting_longlines" then "drifting_longlines"
  when label ="purse_seines" then "purse_seines"
  when label ="other_purse_seines" then "purse_seines"
  when label ="tuna_purse_seines" then "purse_seines"
  when label ="cargo_or_tanker" then "cargo_or_tanker"
  when label ="cargo" then "cargo_or_tanker"
  when label ="tanker" then "cargo_or_tanker"
  when label ="tug" then "tug"
  when label = "trawlers" then "trawlers"
  else "other" end
);

create temp function map_speed(x float64) as (
  case
  when x < 2 then 0
  when x < 4 then 2
  when x < 6 then 4
  when x < 8 then 6
  when x < 10 then 8
  else 10 end
);

create temp function map_minutes(x float64) as (
  case when x < -384 then  -512
  when x < -256 then -384
  when x < -192 then -256
  when x < -128 then -192
  when x < -96 then -128
  when x < -64 then -96
  when x < -48 then -64
  when x < -32 then -48
  when x < -24 then -32
  when x < -16 then -24
  when x < -12 then -16
  when x < -8 then -12
  when x < -6 then -8
  when x < -4 then -6
  when x < -3 then -4
  when x < -2 then -3
  when x < -1 then -2
  when x < 0 then -1
  when x < 1 then 0
  when x < 2 then 1
  when x < 3 then 2
  when x < 4 then 3
  when x < 6 then 4
  when x < 8 then 6
  when x < 12 then 8
  when x < 16 then 12
  when x < 24 then 16
  when x < 32 then 24
  when x < 48 then 32
  when x < 64 then 48
  when x < 96 then 64
  when x < 128 then 96
  when x < 192 then 128
  when x < 256 then 192
  when x < 384 then 256
  else 384 end);

CREATE TEMP FUNCTION reasonable_lon(lon float64) AS
  (case when lon > 180 then lon - 360
  when lon < -180 then lon + 360
  else lon end
);


CREATE TEMP FUNCTION earth_radius_km(lat float64) as
-- This is super overkill. You could just use the average
-- radius of the earth. But I wanted to see if it made a difference.
-- It matters if you want > 4 significant digits.
-- But, once I made it, I didn't feel like deleting it.
-- Equation taken from https://rechneronline.de/earth-radius/
((
select
  --  R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
  pow(
  ( pow(r1*r1 * cos(B),2) + pow(r2*r2 * sin(B),2) )
  /
  ( pow(r1 * cos(B),2) + pow(r2 * sin(B), 2) )
  ,.5)
    from
    (select
    6371.001 as r1,
    6356.752 as r2,
    Radians(lat) as B)
    limit 1
));


with
######################################
-- Data sources
######################################
--
-- vessel_interpolated_positions
-- This is a table created through a previous query that gets the interpolated position
-- of vessels that could be within a given scene. It also includes offset for doppler effect
-- that accounts for satellite direction and speed relative to the vessel

vessel_interpolated_positions as
 (select * from scratch_david.interp_test
-- where ssvid = ssvid()
 )


 ,
--
-- sar detections
sar_detections as (
SELECT TIMESTAMP_ADD(ProductStartTime, INTERVAL cast(timestamp_diff(ProductStopTime, ProductStopTime, SECOND)/2 as int64) SECOND) detect_timestamp,
 DetectionId as detect_id,
 lon detect_lon,
 lat detect_lat,
 scene_id
 FROM proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 -- where DetectionId = '33e8e6ec-01df-4b7d-a907-520ab5a27696'
-- select
-- lat as detect_lat,
-- long as detect_lon,
-- timestamp as detect_timestamp,
-- ACQ as scene_id,
-- CONCAT(ACQ, CAST(lat AS string), CAST(long AS string)) as detect_id
-- from
--  `scratch_bjorn.SAR_detections_april`
--  where date(timestamp) = today()
 ),

--
-- raster probability. "mirror_nozeroes" means that all rows with zero values
-- have been removed, and negative i values have been removed (the raster is
-- symetrical around the i axis because a vessel is just as likely to turn left
-- as it is to turn right).
prob_raster as (select * from
 `world-fishing-827.gfw_research_precursors.point_cloud_mirror_nozeroes_v20190502`
 ),
--
--
-- vessel info table, used for vessel class.
-- acceptable vessel classes are "trawlers","purse_seines","tug","cargo_or_tanker","drifting_longlines","fishing", and "other"
vessel_info_table as (select
        ssvid,
        case when best.best_vessel_class in ("trawlers",
                                            "purse_seines",
                                             "tug","cargo_or_tanker","cargo" , "tanker",
                                             "drifting_longlines") then best.best_vessel_class
        when on_fishing_list_best then "fishing"
        else "other"
        end label,
        from `world-fishing-827.gfw_research.vi_ssvid_v20200410` ),
--
--
##################################
# Probability raster adjustments
##################################
--
-- get the average for general fishing. This is creating a composite category
-- of other categories.
probability_table as (select
   probability,labels,minutes_lower,speed_lower,i,j
from
  prob_raster
union all
select
   avg(probability) probability,"fishing" labels, minutes_lower, speed_lower,i,j
from
  prob_raster
where labels in ("trawlers","purse_seines","drifting_longlines")
group by
   labels,minutes_lower,speed_lower,i,j),
--
--
-- weight is a measure of how concetrated the raster is, and is used to assess score
weights as (
select
  # this is because the value at 0 is not mirrored
  sum(if(i>0, 2*probability *probability, probability *probability )) as weight,
  labels,
  minutes_lower,
  speed_lower
from
  probability_table
group by
  labels,
  minutes_lower,
  speed_lower
),
--
-- combine probabilities and weights into one table
probabilities_and_weights as (
select * from
  probability_table
join
  weights
using(labels, minutes_lower,speed_lower)
),
--
--
-- the raster has only positive i values because it is symmetrical
-- around the i axis (the boat is as equally likely to turn left as right).
-- Make a full raster with negative values for the later join.
probabilities_and_weights_neg as (
select
labels, minutes_lower, speed_lower, probability, i, j, weight
from
probabilities_and_weights
union all
select
-- same except take negative i!
labels, minutes_lower, speed_lower, probability, -i as i, j, weight
from
probabilities_and_weights
where i >0
),
# #########################
# ##
# ## SAR subqueries
# ##
# #########################
# --
# --
# -- start position of the satellite for each image
# start_pos as (
# select
# detect_id,
# timestamp_sub(detect_timestamp, INTERVAL 60 second) as start_time,
# lat as start_lat,
# lon as start_lon,
# altitude as start_altitude
# from sat_positions
# join sar_detections
# on timestamp_sub(timestamp_trunc(detect_timestamp, second), INTERVAL 60 second) = time
# ),


# -- end position of the satellite for each image
# end_pos as (
# select
# detect_id,
# timestamp_add(detect_timestamp, INTERVAL 60 second) as end_time,
# lat as end_lat,
# lon as end_lon,
# altitude as end_altitude
# from sat_positions
# join sar_detections
# on timestamp_add(timestamp_trunc(detect_timestamp, second), INTERVAL 60 second) = time),

# -- calcuate the direction and speed and altitude of the satellite
# deltas as (
# select
# (end_lat - start_lat) * 111 as N_km,
# (end_lon - start_lon) * 111 * cos( radians(end_lat/2 +start_lat/2) ) as E_km,
# end_lat/2 +start_lat/2 as avg_lat,
# start_lat,
# start_lon,
# end_lat,
# end_lon,
# start_altitude,
# end_altitude,
# start_time,
# end_time,
# detect_id
# from end_pos
# join
# start_pos
# using(detect_id)),
# --
# -- What direction is the satellite traveling in each scene?
# sat_directions as (
# select
# detect_id,
# ATAN2(E_Km,N_km)*180/3.1416 sat_course, -- convert to degrees from radians
# start_lat as sat_start_lat,
# start_lon as sat_start_lon,
# start_altitude,
# end_altitude,
# timestamp_diff(end_time, start_time, second) seconds,
# end_lat as sat_end_lat,
# end_lon as sat_end_lon,
# from deltas),
# --
# -- Calculate speed of satellite for each scene
# -- speed of satellite varies a small ammount, so don't really need to calculate
# -- for each scene. But hey -- why not calculate it directly?
# sat_directions_with_speed as (
# select
# st_distance(st_geogpoint(sat_start_lon, sat_start_lat), st_geogpoint(sat_end_lon, sat_end_lat)) -- distance between points in meters
# * (earth_radius_km(sat_end_lat) + start_altitude/1000)/ earth_radius_km(sat_end_lat) -- multiply by a factor to account for satellite altitude
# / seconds  -- make it meters per second
# *1/1852 * 3600 -- * 1 nautical mile / 1852 meters   * 3600 seconds/ hour
# as satellite_knots,
# *
# from sat_directions),
# --
# --
# --
# ############################
# ##
# ## AIS subquery
# ##
# ###########################
# --
# -- lag and lead AIS messages
# ais_messages_lagged_led as (
# select
#   seg_id,
#   lat,
#   lon,
#   timestamp,
#   course,
#   speed_knots as speed,
#   ssvid,
#   distance_from_shore_m,
#   LEAD(timestamp,1) OVER (PARTITION BY seg_id ORDER BY timestamp) timestamp_after,
#   LAG(timestamp,1) OVER (PARTITION BY seg_id ORDER BY timestamp) timestamp_before,
#   date
# from
#   ais_position_table
# where
#   abs(lat) <= 90 and abs(lon) <= 180
#   and seg_id in (select seg_id from good_segs)
#   and speed_knots < 50
#   -- ignore the really fast vessels... most are noise
#   -- this could just ignore speeds of 102.3
# ),
# --
# --
# -- join on image times to get before and after
# best_messages as (
# select
#   a.ssvid ssvid, lat, lon, speed
#   ,course, timestamp,
#   label,
#   distance_from_shore_m,
#   timestamp_diff(detect_timestamp, timestamp, SECOND) / 60.0
#   as delta_minutes,
#   scene_id,
#   detect_timestamp,
#   detect_id,
#   detect_lon,
#   detect_lat,
#   # the following two help a later join. Basically, we want to know if there is another image to join
#   (timestamp_before is not null and abs(timestamp_diff(detect_timestamp, timestamp_before, SECOND)) / 60.0 < 9*60 ) previous_exists,
#   (timestamp_after is not null and abs(timestamp_diff(detect_timestamp, timestamp_after, SECOND)) / 60.0 < 9*60 ) after_exists
# from
#   ais_messages_lagged_led a
# join
#   vessel_info_table b
# on
#   a.ssvid = b.ssvid
# join
#   sar_detections
# on
#   abs(timestamp_diff(detect_timestamp, timestamp, SECOND)) / 60.0  < 9*60 # less than 5 hours
#   and st_distance(st_geogpoint(lon, lat), st_geogpoint(detect_lon, detect_lat)) < 100*1852 -- within 100 nautical miles of the detection
#  -- Timestamps just before or after
#  -- Note that it is really tricky to consider the null value cases, and it makes things a mess later
#  and(
#        (timestamp <= detect_timestamp # timestamp is right before the image
#        AND timestamp_after > detect_timestamp )
#     or (timestamp <= detect_timestamp
#        and timestamp_after is null)
#     or (timestamp > detect_timestamp # timestamp is right after the image
#        AND timestamp_before <= detect_timestamp )
#     or (timestamp > detect_timestamp
#        and timestamp_before is null)
#   )
# ),
# --
# --
# ####################################
# # Figure out adjustment to account for the friggn' doppler shift
# ###################################
# --
# -- Get the interpolated position of each vessel
# -- at the moment of the SAR image
# interpolated_positions as (
# SELECT
#   lat + cos(radians(course)) -- x component
#   *speed*delta_minutes/60 -- nautical miles (knots * minutes / (60 minutes / hour) )
#   / 60  -- divided by the nautical miles per degree lat,
#   -- which is 60 nautical miles per degree (really it is 59.9, and varies by lat)
#   as lat_interpolate,
#   reasonable_lon(
#      lon + sin(radians(course)) -- y component
#      *speed*delta_minutes/60 -- nautical miles (knots * minutes / (60 minutes / hour) )
#      /(60*cos(radians(lat)))) -- divided by the nautical miles per degree lon,
#   -- which is 60 times the cos of the lat (really it is 59.9, and varies by lat)
#   as lon_interpolate,
#   *
# FROM
#   best_messages),
# --
# -- Get distance from the likely position of the vessel to the satellite,
# -- and the speed of the vessel perpendicular to the satellite.
# interpolated_positions_compared_to_satellite as (
# select
#    *,
#    speed * sin(radians( course - sat_course)) as vessel_speed_perpendicular_to_sat,
#    st_distance(safe.st_geogpoint(lon_interpolate, lat_interpolate), -- likely location of vessel
#                ST_MAKELINE( safe.ST_GEOGPOINT(sat_start_lon, sat_start_lat), (safe.ST_GEOGPOINT(sat_end_lon, sat_end_lat) ) ) ) -- line of satellite
#                / 1852 -- convert from meters to nautical miles, because
#                as vessel_distance_to_sat_nm
# from
#   interpolated_positions
# join
#   sat_directions_with_speed
# using(detect_id)
# ),
# --
# -- using satellite speed, vessel speed perpendicular to satellite direction of travel,
# -- and the distance of the vessel to the satellite, calculate the distance the vessel
# -- will be offset in the direction of the satellite is traveling.
# interpolated_positions_adjust_formula as (
# select
#  *,
#  vessel_speed_perpendicular_to_sat / satellite_knots
#  * pow( ( pow(vessel_distance_to_sat_nm,2) + pow(start_altitude,2)/pow(1852,2) ) , .5)
#   -- divide by 1852 to convert meters to nautical miles,
#   -- then use pathangerean theorm to get the approximate distance to the satellite in
#   -- nautical miles.
#   as adjusted_nautical_miles_parallel_to_sat
# from
#   interpolated_positions_compared_to_satellite
# ),
# --
# --
# -- Adjust each lat and lon by the doppler shift. Note the subtraction. If a vessel is traveling
# -- perpendicular to the satellite's motion, going away from the satellite, the vessel will
# -- appear offset parallel to the satellites motion opposite the direction the vessel is traveling.
# -- Believe me! It works!
# best_messages_adjusted as (
# select
#   * except(lon,lat),
#   lat - adjusted_nautical_miles_parallel_to_sat * cos(radians(sat_course))/60 lat, -- 60 nautical miles per degree
#   lon - adjusted_nautical_miles_parallel_to_sat * sin(radians(sat_course))/(60 * cos(radians(lat))) lon, -- 60 nautical miles * cos(lat) per degree
#   lat as old_lat,
#   lon as old_lon
# from
#   interpolated_positions_adjust_formula
# ),

######
#
################


interpolated_unstacked as (
select
scene_id,
ssvid,
timestamp1 as timestamp,
lat1 + lat_doppler_offset1 as lat,
lon1 + lon_doppler_offset1 as lon,
delta_minutes1 as delta_minutes,
speed1 as speed,
scale1 as scale,
course1 as course,
min_lon_raster_lookup1 as min_lon,
min_lat_raster_lookup1 as min_lat,
max_lon_raster_lookup1 as max_lon,
max_lat_raster_lookup1 as max_lat,
lat_interpolate2 is null as is_single
from vessel_interpolated_positions
where lat_interpolate1 is not null
union all
select
scene_id,
ssvid,
timestamp2 as timestamp,
lat2 + lat_doppler_offset2 as lat,
lon2 + lon_doppler_offset2 as lon,
delta_minutes2 as delta_minutes,
speed2 as speed,
scale2 as scale,
course2 as course,
min_lon_raster_lookup2 as min_lon,
min_lat_raster_lookup2 as min_lat,
max_lon_raster_lookup2 as max_lon,
max_lat_raster_lookup2 as max_lat,
lat_interpolate1 is null as is_single
from vessel_interpolated_positions
where lat_interpolate2 is not null
),

# Add vessel class
interpolated_unstacked_w_label as (
    select * from interpolated_unstacked
    join vessel_info_table using(ssvid)
),


detects_vs_interpolated as (
select
ssvid,
label,
scene_id,
detect_id,
detect_lat,
detect_lon,
lat,
lon,
timestamp,
delta_minutes,
speed,
scale,
course,
is_single
from interpolated_unstacked_w_label
join
 sar_detections
 using(scene_id)
where detect_lat between min_lat and max_lat
and detect_lon between min_lon and max_lon

),

--
####################
# joins to the probability raster
###################
--
key_query_1 as (
select *,
  deglat2km() * (detect_lon - lon) * cos(radians(lat)) as u,
  deglat2km() * (detect_lat - lat) as v,
  radians(course) as course_rads
from
  detects_vs_interpolated
),
--
-- rotate the coordinates
key_query_2 as (
select
  *,
  cos(course_rads) * u - sin(course_rads) * v as x,
  cos(course_rads) * v + sin(course_rads) * u as y
  -- rotation of coordinates, described here: https://en.wikipedia.org/wiki/Rotation_of_axes
  -- Note that our u and v / x and y are switched from the standard way to measure
  -- this, largely because vessels measure course from due north, moving clockwise,
  -- while mosth math measures angle from the x axis counterclockwise. Annoying!
  --
#   1000 / colmax(1.0, ABS(delta_minutes)) as scale
#     This is the python function we are copying here:
#      def scale(dt):
#         return 1000.0 / max(1, abs(dt))
from
  key_query_1
),
--
-- adjust by scale -- that is the probability raster will be scalled
-- based on how long before or after the ping the image was taken.
-- Also, move the raster so that 0,0 is where the vessel would be
-- if it traveled in a straight line.
key_query_3 as
(
select * from (
  select *,
    x * scale as x_key,
    (y - speed*kilometers_per_nautical_mile()*delta_minutes/60 ) * scale  as y_key, -- the old version interpolated, and this one doesn't
    # y * scale as y_key,
    # Map these values to the values in the probability rasters
    map_speed(speed) as speed_key,
    map_minutes(delta_minutes) as minute_key,
    map_label(label) as label_key
  from
    key_query_2
  )
where abs(x_key) <=500 and abs(y_key) <=500
),
--
--
-- Match to probability, and interpolate between
-- the four closest values. This bilinear interpoloation
-- in theory allows us to reduce the size of the raster we are joining on
messages_with_probabilities as
(
select
  -- this would get the value exact, the weight_scaled
  -- / pow((1000/(colmax( 1, probs.minutes_lower/2 + probs.minutes_upper /2))),2) * scale*scale
  * except(i, j, probability),
  bilinear_interpolation(
  ifnull(probs_11.probability,0),
  ifnull(probs_12.probability,0),
  ifnull(probs_22.probability,0),
  ifnull(probs_21.probability,0),
  cast(x_key - .5 as int64), cast(x_key + .5 as int64),
  cast(y_key - .5 as int64), cast(y_key + .5 as int64) ,
  x_key, y_key) as probability,
  -- to get at least one value.
  -- weight *should* be the same for each, but we need to make sure it isn't null
  case when probs_11.weight is not null then probs_11.weight/(scale*scale)
  when probs_12.weight is not null then probs_12.weight/(scale*scale)
  when probs_22.weight is not null then probs_22.weight/(scale*scale)
  when probs_21.weight is not null then probs_21.weight/(scale*scale)
  else 0
  end
  as weight_scaled
from
  key_query_3
left join
  --
  -- joining on four locaitons to do bilinear interpolation
probabilities_and_weights_neg  as probs_11
  on  probs_11.i = cast(x_key - .5 as int64) and probs_11.j = cast(y_key - .5 as int64)
  and probs_11.speed_lower = cast(speed_key as int64)
  and probs_11.minutes_lower = cast(minute_key as int64)
  and probs_11.labels = label_key
left join
probabilities_and_weights_neg  as probs_12
  on  probs_12.i = cast(x_key -.5 as int64) and probs_12.j = cast(y_key + .5 as int64)
  and probs_12.speed_lower = cast(speed_key as int64)
  and probs_12.minutes_lower = cast(minute_key as int64)
  and probs_12.labels = label_key
left join
probabilities_and_weights_neg  as probs_22
  on  probs_22.i = cast(x_key +.5 as int64) and probs_22.j = cast(y_key + .5 as int64)
  and probs_22.speed_lower = cast(speed_key as int64)
  and probs_22.minutes_lower = cast(minute_key as int64)
  and probs_22.labels = label_key
left join
probabilities_and_weights_neg  as probs_21
  on  probs_21.i = cast(x_key +.5 as int64) and probs_21.j = cast(y_key - .5 as int64)
  and probs_21.speed_lower = cast(speed_key as int64)
  and probs_21.minutes_lower = cast(minute_key as int64)
  and probs_21.labels = label_key
),
--
--
--
joined_detects AS (
SELECT
  a.ssvid,
  a.scene_id scene_id,
  a.detect_id detect_id,
  if(a.probability is null, 0, a.probability ) a_probability,
  if(b.probability is null, 0, b.probability ) b_probability,
  if(a.weight_scaled is null, 0, a.weight_scaled ) a_weight_scaled,
  if(b.weight_scaled is null, 0, b.weight_scaled ) b_weight_scaled,
  a.scale a_scale,
  b.scale b_scale,
  a.detect_lat detect_lat,
  a.detect_lon detect_lon,
  a.timestamp a_timestamp,
  b.timestamp b_timestamp,
  a.speed a_speed,
  b.speed b_speed,
  a.course a_course,
  b.course b_course,
  a.lat a_lat,
  a.lon a_lon,
  b.lat b_lat,
  b.lon b_lon,
#   a.detect_timestamp detect_timestamp
FROM
  messages_with_probabilities a
left join
  messages_with_probabilities b
ON
  a.ssvid = b.ssvid
  AND a.detect_id = b.detect_id
  AND a.timestamp > b.timestamp
  # the following makes sure that a message isn't included that shouldn't be
  # Basically, the left join includes things that actually join so the point gets repeated,
  # but with a null value. This fixes that problem
where not ( b.timestamp is null and (not a.is_single or not b.is_single))
),
--
-- Apply a score to each detection to vessel
-- This score was figured out by Tim Hochberg, who tried
-- out a series of different things.
 scored_detects as (
select *,
   safe_divide( (a_probability*a_weight_scaled +
    b_probability*b_weight_scaled),
a_weight_scaled + b_weight_scaled) AS score
from joined_detects)
--
--
#################################################
# And the answer is....
#################################################
--

# select * from messages_with_probabilities
# order by detect_id, ssvid, minute_key

select "ais" as source, * from scored_detects where score > 0)
"""


# +
# prob_nomed

q = """#StandardSql

CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
y1 float64, y2 float64, x float64, y float64) AS (
    # see https://en.wikipedia.org/wiki/Bilinear_interpolation
    # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
    # x and y are the coordinates we want the value for
    1/( (x2 - x1) * (y2 - y1)) *
      (
         Q11 * (x2 - x) * (y2 - y)
       + Q21 * (x - x1) * (y2 - y)
       + Q12 * (x2 - x) * (y - y1)
       + Q22 * (x - x1) * (y - y1)
      )
    );


create temp function radians(x float64) as (
  3.14159265359 * x / 180
);

create temp function deglat2km() as (
  111.699
);

create temp function meters2deg() as (
   1/(deglat2km()*1000)
);

create temp function kilometers_per_nautical_mile() as (
  1.852
);

create temp function one_over_cellsize() as (
  200  -- 500 meter resolution roughly
);

create temp function map_label(label string)
as (
  case when label ="drifting_longlines" then "drifting_longlines"
  when label ="purse_seines" then "purse_seines"
  when label ="other_purse_seines" then "purse_seines"
  when label ="tuna_purse_seines" then "purse_seines"
  when label ="cargo_or_tanker" then "cargo_or_tanker"
  when label ="cargo" then "cargo_or_tanker"
  when label ="tanker" then "cargo_or_tanker"
  when label ="tug" then "tug"
  when label = "trawlers" then "trawlers"
  else "other" end
);

create temp function map_speed(x float64) as (
  case
  when x < 2 then 0
  when x < 4 then 2
  when x < 6 then 4
  when x < 8 then 6
  when x < 10 then 8
  else 10 end
);

create temp function map_minutes(x float64) as (
  case when x < -384 then  -512
  when x < -256 then -384
  when x < -192 then -256
  when x < -128 then -192
  when x < -96 then -128
  when x < -64 then -96
  when x < -48 then -64
  when x < -32 then -48
  when x < -24 then -32
  when x < -16 then -24
  when x < -12 then -16
  when x < -8 then -12
  when x < -6 then -8
  when x < -4 then -6
  when x < -3 then -4
  when x < -2 then -3
  when x < -1 then -2
  when x < 0 then -1
  when x < 1 then 0
  when x < 2 then 1
  when x < 3 then 2
  when x < 4 then 3
  when x < 6 then 4
  when x < 8 then 6
  when x < 12 then 8
  when x < 16 then 12
  when x < 24 then 16
  when x < 32 then 24
  when x < 48 then 32
  when x < 64 then 48
  when x < 96 then 64
  when x < 128 then 96
  when x < 192 then 128
  when x < 256 then 192
  when x < 384 then 256
  else 384 end);

CREATE TEMP FUNCTION reasonable_lon(lon float64) AS
  (case when lon > 180 then lon - 360
  when lon < -180 then lon + 360
  else lon end
);


CREATE TEMP FUNCTION earth_radius_km(lat float64) as
-- this is super overkill. You could just use the average
-- radius of the earth. But I wanted to see if it made a difference.
-- It matters if you want > 4 significant digits.
-- But, once I made it, I didn't feel like deleting it.
-- Equation taken from https://rechneronline.de/earth-radius/
((
select
  --  R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
  pow(
  ( pow(r1*r1 * cos(B),2) + pow(r2*r2 * sin(B),2) )
  /
  ( pow(r1 * cos(B),2) + pow(r2 * sin(B), 2) )
  ,.5)
    from
    (select
    6371.001 as r1,
    6356.752 as r2,
    Radians(lat) as B)
    limit 1
));


with
######################################
-- Data sources
######################################
--
-- interpolated positions
-- vessel info table, used for vessel class.
-- acceptable vessel classes are "trawlers","purse_seines","tug","cargo_or_tanker","drifting_longlines","fishing", and "other"
vessel_info_table as (select
        ssvid,
        case when best.best_vessel_class in ("trawlers",
                                            "purse_seines",
                                             "tug","cargo_or_tanker",
                                             "tanker","cargo",
                                             "drifting_longlines") then best.best_vessel_class
        when on_fishing_list_best then "fishing"
        -- when best.best_vessel_class = "gear" then "gear"
        else "other"
        end label,
        best.best_vessel_class = 'gear' is_gear
        from `world-fishing-827.gfw_research.vi_ssvid_v20200410` ),

interp_positions as (select * except(label),
                        ifnull(label, "other") label ,
                        case when min_lon_raster_lookup2 is null then min_lon_raster_lookup1
                        when min_lon_raster_lookup1 is null then min_lon_raster_lookup2
                        else least(min_lon_raster_lookup1,min_lon_raster_lookup2) end min_lon,
                        case when min_lat_raster_lookup2 is null then min_lat_raster_lookup1
                        when min_lat_raster_lookup1 is null then min_lat_raster_lookup2
                        else least(min_lat_raster_lookup1,min_lat_raster_lookup2) end min_lat,
                        case when max_lon_raster_lookup2 is null then max_lon_raster_lookup1
                        when max_lon_raster_lookup1 is null then max_lon_raster_lookup2
                        else greatest(max_lon_raster_lookup1,max_lon_raster_lookup2) end max_lon,
                        case when max_lat_raster_lookup2 is null then max_lat_raster_lookup1
                        when max_lat_raster_lookup1 is null then max_lat_raster_lookup2
                        else greatest(max_lat_raster_lookup1,max_lat_raster_lookup2) end max_lat,
                    from scratch_david.interp_test
                      left join vessel_info_table
                      using(ssvid)),


-- sar detections
scene_footprints as (
SELECT
distinct
TIMESTAMP_ADD(ProductStartTime, INTERVAL
cast(timestamp_diff(ProductStopTime, ProductStopTime,
SECOND)/2 as int64) SECOND) scene_timestamp,
scene_id,
footprint
FROM (select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
union all
select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
)
 ),

--
-- raster probability. "mirror_nozeroes" means that all rows with zero values
-- have been removed, and negative i values have been removed (the raster is
-- symetrical around the i axis because a vessel is just as likely to turn left
-- as it is to turn right).
prob_raster as (select * from
 `world-fishing-827.gfw_research_precursors.point_cloud_mirror_nozeroes_v20190502`
 ),
--

--

--
--
##################################
# Probability raster adjustments
##################################
--
-- get the average for general fishing. This is creating a composite category
-- of other categories.
probability_table as (select
   probability,labels,minutes_lower,speed_lower,i,j
from
  prob_raster
union all
select
   avg(probability) probability,"fishing" labels, minutes_lower, speed_lower,i,j
from
  prob_raster
where labels in ("trawlers","purse_seines","drifting_longlines")
group by
   labels,minutes_lower,speed_lower,i,j),
--
--

--
--
-- the raster has only positive i values because it is symmetrical
-- around the i axis (the boat is as equally likely to turn left as right).
-- Make a full raster with negative values for the later join.
probabilities_and_weights_neg as (
select
labels, minutes_lower, speed_lower, probability, i, j --, weight
from
probability_table
union all
select
-- same except take negative i!
labels, minutes_lower, speed_lower, probability, -i as i, j -- , weight
from
probability_table
where i >0
),
#########################
##
## SAR subqueries
##
#########################
--
--
just_bounds as
(select
ssvid,
label,
scene_id,
speed1 as speed,
course1 as course,
scale1 as scale,
delta_minutes2 is null as is_single,
delta_minutes1 as delta_minutes,
# least(min_lon_raster_lookup1, ifnull(min_lon_raster_lookup2,min_lon_raster_lookup1)) as min_lon,
# greatest(max_lon_raster_lookup1, ifnull(max_lon_raster_lookup2,max_lon_raster_lookup1)) as max_lon,
# least(min_lat_raster_lookup1, ifnull(min_lat_raster_lookup2,min_lat_raster_lookup1)) as min_lat,
# greatest(max_lat_raster_lookup1, ifnull(max_lat_raster_lookup2,max_lat_raster_lookup1)) as max_lat,
min_lon,
max_lon,
min_lat,
max_lat,
lat_interpolate1 + lat_doppler_offset1 as lat_interpolate_adjusted,
lon_interpolate1 + lon_doppler_offset1 as lon_interpolate_adjusted
from interp_positions where delta_minutes1 is not null
union all
select
ssvid,
label,
scene_id,
speed2 as speed,
course2 as course,
scale2 as scale,
delta_minutes1 is null as is_single,
delta_minutes2 as delta_minutes,
# least(min_lon_raster_lookup2, ifnull(min_lon_raster_lookup1,min_lon_raster_lookup2)) as min_lon,
# greatest(max_lon_raster_lookup2, ifnull(max_lon_raster_lookup1,max_lon_raster_lookup2)) as max_lon,
# least(min_lat_raster_lookup2, ifnull(min_lat_raster_lookup1,min_lat_raster_lookup2)) as min_lat,
# greatest(max_lat_raster_lookup2, ifnull(max_lat_raster_lookup1,max_lat_raster_lookup2)) as max_lat,
min_lon,
max_lon,
min_lat,
max_lat,
lat_interpolate2 + lat_doppler_offset2 as lat_interpolate_adjusted,
lon_interpolate2 + lon_doppler_offset2 as lon_interpolate_adjusted
from interp_positions where delta_minutes2 is not null
),

 lat_array AS(
  SELECT
    * ,
    lat_bin/one_over_cellsize() as detect_lat  -- to get the middle of the cell
  FROM
    just_bounds
  CROSS JOIN
    UNNEST(GENERATE_ARRAY(
           cast(FLOOR(min_lat*one_over_cellsize()) as int64) - 2, -- just to pad it a bit
           cast(FLOOR(max_lat*one_over_cellsize()) as int64) +2,
           1))
     AS lat_bin),


  lon_array AS (
  SELECT
    *,
    lon_bin/one_over_cellsize() as detect_lon -- to get the middle of the cell
  FROM
    just_bounds
  CROSS JOIN
        UNNEST(GENERATE_ARRAY(
           cast(FLOOR(min_lon*one_over_cellsize()) as int64) - 2, -- just to pad it a bit
           cast(FLOOR(max_lon*one_over_cellsize()) as int64) + 2,
           1))  AS lon_bin),



  id_lat_lon_array AS (
  select
  a.ssvid,
  a.label,
  a.scene_id,
  a.speed,
  a.course,
  a.scale,
  a.is_single,
  a.delta_minutes,
  a.lat_interpolate_adjusted,
  a.lon_interpolate_adjusted,
  lon_bin,
  lat_bin,
  detect_lon,
  detect_lat,
  FROM
    lon_array a
  CROSS JOIN
    lat_array b
  WHERE
    a.scene_id=b.scene_id
    and a.ssvid=b.ssvid
    and a.delta_minutes = b.delta_minutes),

--
####################
# joins to the probability raster
###################
--

key_query_1 as (
select *,
  deglat2km() * (detect_lon - lon_interpolate_adjusted) * cos(radians(lat_interpolate_adjusted)) as u,
  deglat2km() * (detect_lat - lat_interpolate_adjusted) as v,
  radians(course) as course_rads
from
  id_lat_lon_array
),

# --
# -- rotate the coordinates
key_query_2 as (
select
  *,
  cos(course_rads) * u - sin(course_rads) * v as x,
  cos(course_rads) * v + sin(course_rads) * u as y,
  -- rotation of coordinates, described here: https://en.wikipedia.org/wiki/Rotation_of_axes
  -- Note that our u and v / x and y are switched from the standard way to measure
  -- this, largely because vessels measure course from due north, moving clockwise,
  -- while mosth math measures angle from the x axis counterclockwise. Annoying!
  --
  # 1000 / colmax(1.0, ABS(delta_minutes)) as scale
#     This is the python function we are copying here:
#      def scale(dt):
#         return 1000.0 / max(1, abs(dt))
from
  key_query_1
),
# --
# -- adjust by scale -- that is the probability raster will be scalled
# -- based on how long before or after the ping the image was taken.
# -- Also, move the raster so that 0,0 is where the vessel would be
# -- if it traveled in a straight line.
key_query_3 as
(
select * from (
  select *,
    x * scale as x_key,
    # (y - speed*kilometers_per_nautical_mile()*delta_minutes/60 ) * scale  as y_key,
    y  * scale  as y_key, # because using interpolated position, already centered at this location
    # Map these values to the values in the probability rasters
    map_speed(speed) as speed_key,
    map_minutes(delta_minutes) as minute_key,
    map_label(label) as label_key
  from
    key_query_2
  )
where abs(x_key) <=500 and abs(y_key) <=500
),



# # --
# # --
-- Match to probability, and interpolate between
-- the four closest values. This bilinear interpoloation
-- in theory allows us to reduce the size of the raster we are joining on
messages_with_probabilities as
 (
 select
--   -- this would get the value exact, the weight_scaled
--   -- / pow((1000/(colmax( 1, probs.minutes_lower/2 + probs.minutes_upper /2))),2) * scale*scale
   * except(i, j, probability),
   bilinear_interpolation(
   ifnull(probs_11.probability,0),
   ifnull(probs_12.probability,0),
   ifnull(probs_22.probability,0),
   ifnull(probs_21.probability,0),
   cast(x_key - .5 as int64), cast(x_key + .5 as int64),
   cast(y_key - .5 as int64), cast(y_key + .5 as int64) ,
   x_key, y_key) as probability,
--   -- to get at least one value.
--   -- weight *should* be the same for each, but we need to make sure it isn't null
 from
   key_query_3
 left join
-- joining on four locaitons to do bilinear interpolation
 probabilities_and_weights_neg  as probs_11
   on  probs_11.i = cast(x_key - .5 as int64) and probs_11.j = cast(y_key - .5 as int64)
   and probs_11.speed_lower = cast(speed_key as int64)
   and probs_11.minutes_lower = cast(minute_key as int64)
   and probs_11.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_12
   on  probs_12.i = cast(x_key -.5 as int64) and probs_12.j = cast(y_key + .5 as int64)
   and probs_12.speed_lower = cast(speed_key as int64)
   and probs_12.minutes_lower = cast(minute_key as int64)
   and probs_12.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_22
   on  probs_22.i = cast(x_key +.5 as int64) and probs_22.j = cast(y_key + .5 as int64)
   and probs_22.speed_lower = cast(speed_key as int64)
   and probs_22.minutes_lower = cast(minute_key as int64)
   and probs_22.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_21
   on  probs_21.i = cast(x_key +.5 as int64) and probs_21.j = cast(y_key - .5 as int64)
   and probs_21.speed_lower = cast(speed_key as int64)
   and probs_21.minutes_lower = cast(minute_key as int64)
   and probs_21.labels = label_key
 ),




prob_multiplied_table as
(
  select
  a.ssvid ,
  a.scene_id,
  a.probability*b.probability probability,
  a.detect_lat,
  a.detect_lon
  from messages_with_probabilities a
  join
  messages_with_probabilities b
  on a.ssvid = b.ssvid
  and a.lat_bin = b.lat_bin
  and a.lon_bin = b.lon_bin
  and b.delta_minutes < a.delta_minutes
  and not a.is_single
  and a.probability > 0 and b.probability > 0
  union all
  select
  ssvid,
  scene_id,
  probability,
  detect_lat,
  detect_lon
  from
  messages_with_probabilities
  where is_single
)

select *, probability/prob_sum probability_norm from prob_multiplied_table
join
(select ssvid, scene_id, sum(probability) prob_sum from prob_multiplied_table
 group by ssvid, scene_id)
 using(ssvid,scene_id)





# union all




# # select
# # ssvid,
# # scene_id,
# # seg_id,
# # detect_lat,
# # detect_lon,
# # old_lat,
# # old_lon,
# # lat_interpolate_adjusted,
# # lon_interpolate_adjusted,
# # label,
# # speed,
# # timestamp,
# # probability,
# # scale,
# # delta_minutes,
# #  from  messages_with_probabilities
# # where probability > 0




# select  * from scratch_david.interp_test"""


with open("temp.sql", "w") as f:
    f.write(q)

command = """cat temp.sql | bq query --replace \
        --destination_table=scratch_david.interp_raster_norm_test\
         --allow_large_results --use_legacy_sql=false"""
os.system(command)

os.system("rm -f temp.sql")

# +

q = """#StandardSql

CREATE TEMP FUNCTION bilinear_interpolation(Q11 float64,
Q12 float64, Q22 float64, Q21 float64, x1 float64, x2 float64,
y1 float64, y2 float64, x float64, y float64) AS (
    # see https://en.wikipedia.org/wiki/Bilinear_interpolation
    # Q11 is the value at x1, y1, Q12 is the value at x1, y2, etc.
    # x and y are the coordinates we want the value for
    1/( (x2 - x1) * (y2 - y1)) *
      (
         Q11 * (x2 - x) * (y2 - y)
       + Q21 * (x - x1) * (y2 - y)
       + Q12 * (x2 - x) * (y - y1)
       + Q22 * (x - x1) * (y - y1)
      )
    );


create temp function radians(x float64) as (
  3.14159265359 * x / 180
);

create temp function deglat2km() as (
  111.699
);

create temp function meters2deg() as (
   1/(deglat2km()*1000)
);

create temp function kilometers_per_nautical_mile() as (
  1.852
);

create temp function one_over_cellsize() as (
  200  -- 500 meter resolution roughly
);

create temp function map_label(label string)
as (
  case when label ="drifting_longlines" then "drifting_longlines"
  when label ="purse_seines" then "purse_seines"
  when label ="other_purse_seines" then "purse_seines"
  when label ="tuna_purse_seines" then "purse_seines"
  when label ="cargo_or_tanker" then "cargo_or_tanker"
  when label ="cargo" then "cargo_or_tanker"
  when label ="tanker" then "cargo_or_tanker"
  when label ="tug" then "tug"
  when label = "trawlers" then "trawlers"
  else "other" end
);

create temp function map_speed(x float64) as (
  case
  when x < 2 then 0
  when x < 4 then 2
  when x < 6 then 4
  when x < 8 then 6
  when x < 10 then 8
  else 10 end
);

create temp function map_minutes(x float64) as (
  case when x < -384 then  -512
  when x < -256 then -384
  when x < -192 then -256
  when x < -128 then -192
  when x < -96 then -128
  when x < -64 then -96
  when x < -48 then -64
  when x < -32 then -48
  when x < -24 then -32
  when x < -16 then -24
  when x < -12 then -16
  when x < -8 then -12
  when x < -6 then -8
  when x < -4 then -6
  when x < -3 then -4
  when x < -2 then -3
  when x < -1 then -2
  when x < 0 then -1
  when x < 1 then 0
  when x < 2 then 1
  when x < 3 then 2
  when x < 4 then 3
  when x < 6 then 4
  when x < 8 then 6
  when x < 12 then 8
  when x < 16 then 12
  when x < 24 then 16
  when x < 32 then 24
  when x < 48 then 32
  when x < 64 then 48
  when x < 96 then 64
  when x < 128 then 96
  when x < 192 then 128
  when x < 256 then 192
  when x < 384 then 256
  else 384 end);

CREATE TEMP FUNCTION reasonable_lon(lon float64) AS
  (case when lon > 180 then lon - 360
  when lon < -180 then lon + 360
  else lon end
);


CREATE TEMP FUNCTION earth_radius_km(lat float64) as
-- this is super overkill. You could just use the average
-- radius of the earth. But I wanted to see if it made a difference.
-- It matters if you want > 4 significant digits.
-- But, once I made it, I didn't feel like deleting it.
-- Equation taken from https://rechneronline.de/earth-radius/
((
select
  --  R = √ [ (r1² * cos(B))² + (r2² * sin(B))² ] / [ (r1 * cos(B))² + (r2 * sin(B))² ]
  pow(
  ( pow(r1*r1 * cos(B),2) + pow(r2*r2 * sin(B),2) )
  /
  ( pow(r1 * cos(B),2) + pow(r2 * sin(B), 2) )
  ,.5)
    from
    (select
    6371.001 as r1,
    6356.752 as r2,
    Radians(lat) as B)
    limit 1
));


with
######################################
-- Data sources
######################################
--
-- interpolated positions
-- vessel info table, used for vessel class.
-- acceptable vessel classes are "trawlers","purse_seines","tug","cargo_or_tanker","drifting_longlines","fishing", and "other"
vessel_info_table as (select
        ssvid,
        case when best.best_vessel_class in ("trawlers",
                                            "purse_seines",
                                             "tug","cargo_or_tanker",
                                             "tanker","cargo",
                                             "drifting_longlines") then best.best_vessel_class
        when on_fishing_list_best then "fishing"
        -- when best.best_vessel_class = "gear" then "gear"
        else "other"
        end label,
        best.best_vessel_class = 'gear' is_gear
        from `world-fishing-827.gfw_research.vi_ssvid_v20200410` ),

interp_positions as (select * except(label),
                        ifnull(label, "other") label ,
                        case when min_lon_raster_lookup2 is null then min_lon_raster_lookup1
                        when min_lon_raster_lookup1 is null then min_lon_raster_lookup2
                        else least(min_lon_raster_lookup1,min_lon_raster_lookup2) end min_lon,
                        case when min_lat_raster_lookup2 is null then min_lat_raster_lookup1
                        when min_lat_raster_lookup1 is null then min_lat_raster_lookup2
                        else least(min_lat_raster_lookup1,min_lat_raster_lookup2) end min_lat,
                        case when max_lon_raster_lookup2 is null then max_lon_raster_lookup1
                        when max_lon_raster_lookup1 is null then max_lon_raster_lookup2
                        else greatest(max_lon_raster_lookup1,max_lon_raster_lookup2) end max_lon,
                        case when max_lat_raster_lookup2 is null then max_lat_raster_lookup1
                        when max_lat_raster_lookup1 is null then max_lat_raster_lookup2
                        else greatest(max_lat_raster_lookup1,max_lat_raster_lookup2) end max_lat,
                    from scratch_david.interp_test
                      left join vessel_info_table
                      using(ssvid)),


-- sar detections
scene_footprints as (
SELECT
distinct
TIMESTAMP_ADD(ProductStartTime, INTERVAL
cast(timestamp_diff(ProductStopTime, ProductStopTime,
SECOND)/2 as int64) SECOND) scene_timestamp,
scene_id,
footprint
FROM (select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
union all
select ProductStartTime, ProductStopTime,scene_id,footprint from
proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110
)
 ),

--
-- raster probability. "mirror_nozeroes" means that all rows with zero values
-- have been removed, and negative i values have been removed (the raster is
-- symetrical around the i axis because a vessel is just as likely to turn left
-- as it is to turn right).
prob_raster as (select * from
 `world-fishing-827.gfw_research_precursors.point_cloud_mirror_nozeroes_v20190502`
 ),
--

--

--
--
##################################
# Probability raster adjustments
##################################
--
-- get the average for general fishing. This is creating a composite category
-- of other categories.
probability_table as (select
   probability,labels,minutes_lower,speed_lower,i,j
from
  prob_raster
union all
select
   avg(probability) probability,"fishing" labels, minutes_lower, speed_lower,i,j
from
  prob_raster
where labels in ("trawlers","purse_seines","drifting_longlines")
group by
   labels,minutes_lower,speed_lower,i,j),
--
--

--
--
-- the raster has only positive i values because it is symmetrical
-- around the i axis (the boat is as equally likely to turn left as right).
-- Make a full raster with negative values for the later join.
probabilities_and_weights_neg as (
select
labels, minutes_lower, speed_lower, probability, i, j --, weight
from
probability_table
union all
select
-- same except take negative i!
labels, minutes_lower, speed_lower, probability, -i as i, j -- , weight
from
probability_table
where i >0
),
#########################
##
## SAR subqueries
##
#########################
--
--
just_bounds as
(select
ssvid,
label,
scene_id,
speed1 as speed,
course1 as course,
scale1 as scale,
delta_minutes2 is null as is_single,
delta_minutes1 as delta_minutes,
# least(min_lon_raster_lookup1, ifnull(min_lon_raster_lookup2,min_lon_raster_lookup1)) as min_lon,
# greatest(max_lon_raster_lookup1, ifnull(max_lon_raster_lookup2,max_lon_raster_lookup1)) as max_lon,
# least(min_lat_raster_lookup1, ifnull(min_lat_raster_lookup2,min_lat_raster_lookup1)) as min_lat,
# greatest(max_lat_raster_lookup1, ifnull(max_lat_raster_lookup2,max_lat_raster_lookup1)) as max_lat,
min_lon,
max_lon,
min_lat,
max_lat,
lat_interpolate1 + lat_doppler_offset1 as lat_interpolate_adjusted,
lon_interpolate1 + lon_doppler_offset1 as lon_interpolate_adjusted
from interp_positions where delta_minutes1 is not null
union all
select
ssvid,
label,
scene_id,
speed2 as speed,
course2 as course,
scale2 as scale,
delta_minutes1 is null as is_single,
delta_minutes2 as delta_minutes,
# least(min_lon_raster_lookup2, ifnull(min_lon_raster_lookup1,min_lon_raster_lookup2)) as min_lon,
# greatest(max_lon_raster_lookup2, ifnull(max_lon_raster_lookup1,max_lon_raster_lookup2)) as max_lon,
# least(min_lat_raster_lookup2, ifnull(min_lat_raster_lookup1,min_lat_raster_lookup2)) as min_lat,
# greatest(max_lat_raster_lookup2, ifnull(max_lat_raster_lookup1,max_lat_raster_lookup2)) as max_lat,
min_lon,
max_lon,
min_lat,
max_lat,
lat_interpolate2 + lat_doppler_offset2 as lat_interpolate_adjusted,
lon_interpolate2 + lon_doppler_offset2 as lon_interpolate_adjusted
from interp_positions where delta_minutes2 is not null
),

 lat_array AS(
  SELECT
    * ,
    lat_bin/one_over_cellsize() as detect_lat  -- to get the middle of the cell
  FROM
    just_bounds
  CROSS JOIN
    UNNEST(GENERATE_ARRAY(
           cast(FLOOR(min_lat*one_over_cellsize()) as int64) - 2, -- just to pad it a bit
           cast(FLOOR(max_lat*one_over_cellsize()) as int64) +2,
           1))
     AS lat_bin),


  lon_array AS (
  SELECT
    *,
    lon_bin/one_over_cellsize() as detect_lon -- to get the middle of the cell
  FROM
    just_bounds
  CROSS JOIN
        UNNEST(GENERATE_ARRAY(
           cast(FLOOR(min_lon*one_over_cellsize()) as int64) - 2, -- just to pad it a bit
           cast(FLOOR(max_lon*one_over_cellsize()) as int64) + 2,
           1))  AS lon_bin),



  id_lat_lon_array AS (
  select
  a.ssvid,
  a.label,
  a.scene_id,
  a.speed,
  a.course,
  a.scale,
  a.is_single,
  a.delta_minutes,
  a.lat_interpolate_adjusted,
  a.lon_interpolate_adjusted,
  lon_bin,
  lat_bin,
  detect_lon,
  detect_lat,
  FROM
    lon_array a
  CROSS JOIN
    lat_array b
  WHERE
    a.scene_id=b.scene_id
    and a.ssvid=b.ssvid
    and a.delta_minutes = b.delta_minutes),

--
####################
# joins to the probability raster
###################
--

key_query_1 as (
select *,
  deglat2km() * (detect_lon - lon_interpolate_adjusted) * cos(radians(lat_interpolate_adjusted)) as u,
  deglat2km() * (detect_lat - lat_interpolate_adjusted) as v,
  radians(course) as course_rads
from
  id_lat_lon_array
),

# --
# -- rotate the coordinates
key_query_2 as (
select
  *,
  cos(course_rads) * u - sin(course_rads) * v as x,
  cos(course_rads) * v + sin(course_rads) * u as y,
  -- rotation of coordinates, described here: https://en.wikipedia.org/wiki/Rotation_of_axes
  -- Note that our u and v / x and y are switched from the standard way to measure
  -- this, largely because vessels measure course from due north, moving clockwise,
  -- while mosth math measures angle from the x axis counterclockwise. Annoying!
  --
  # 1000 / colmax(1.0, ABS(delta_minutes)) as scale
#     This is the python function we are copying here:
#      def scale(dt):
#         return 1000.0 / max(1, abs(dt))
from
  key_query_1
),
# --
# -- adjust by scale -- that is the probability raster will be scalled
# -- based on how long before or after the ping the image was taken.
# -- Also, move the raster so that 0,0 is where the vessel would be
# -- if it traveled in a straight line.
key_query_3 as
(
select * from (
  select *,
    x * scale as x_key,
    # (y - speed*kilometers_per_nautical_mile()*delta_minutes/60 ) * scale  as y_key,
    y  * scale  as y_key, # because using interpolated position, already centered at this location
    # Map these values to the values in the probability rasters
    map_speed(speed) as speed_key,
    map_minutes(delta_minutes) as minute_key,
    map_label(label) as label_key
  from
    key_query_2
  )
where abs(x_key) <=500 and abs(y_key) <=500
),



# # --
# # --
-- Match to probability, and interpolate between
-- the four closest values. This bilinear interpoloation
-- in theory allows us to reduce the size of the raster we are joining on
messages_with_probabilities as
 (
 select
--   -- this would get the value exact, the weight_scaled
--   -- / pow((1000/(colmax( 1, probs.minutes_lower/2 + probs.minutes_upper /2))),2) * scale*scale
   * except(i, j, probability),
   bilinear_interpolation(
   ifnull(probs_11.probability,0),
   ifnull(probs_12.probability,0),
   ifnull(probs_22.probability,0),
   ifnull(probs_21.probability,0),
   cast(x_key - .5 as int64), cast(x_key + .5 as int64),
   cast(y_key - .5 as int64), cast(y_key + .5 as int64) ,
   x_key, y_key) as probability,
--   -- to get at least one value.
--   -- weight *should* be the same for each, but we need to make sure it isn't null
 from
   key_query_3
 left join
-- joining on four locaitons to do bilinear interpolation
 probabilities_and_weights_neg  as probs_11
   on  probs_11.i = cast(x_key - .5 as int64) and probs_11.j = cast(y_key - .5 as int64)
   and probs_11.speed_lower = cast(speed_key as int64)
   and probs_11.minutes_lower = cast(minute_key as int64)
   and probs_11.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_12
   on  probs_12.i = cast(x_key -.5 as int64) and probs_12.j = cast(y_key + .5 as int64)
   and probs_12.speed_lower = cast(speed_key as int64)
   and probs_12.minutes_lower = cast(minute_key as int64)
   and probs_12.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_22
   on  probs_22.i = cast(x_key +.5 as int64) and probs_22.j = cast(y_key + .5 as int64)
   and probs_22.speed_lower = cast(speed_key as int64)
   and probs_22.minutes_lower = cast(minute_key as int64)
   and probs_22.labels = label_key
 left join
 probabilities_and_weights_neg  as probs_21
   on  probs_21.i = cast(x_key +.5 as int64) and probs_21.j = cast(y_key - .5 as int64)
   and probs_21.speed_lower = cast(speed_key as int64)
   and probs_21.minutes_lower = cast(minute_key as int64)
   and probs_21.labels = label_key
 ),




prob_multiplied_table as
(
  select
  a.ssvid ,
  a.scene_id,
  a.probability*b.probability probability,
  a.detect_lat,
  a.detect_lon
  from messages_with_probabilities a
  join
  messages_with_probabilities b
  on a.ssvid = b.ssvid
  and a.lat_bin = b.lat_bin
  and a.lon_bin = b.lon_bin
  and b.delta_minutes < a.delta_minutes
  and not a.is_single
  and a.probability > 0 and b.probability > 0
  union all
  select
  ssvid,
  scene_id,
  probability,
  detect_lat,
  detect_lon
  from
  messages_with_probabilities
  where is_single
),

normalized_prob as (
select *, probability/prob_sum probability_norm from prob_multiplied_table
join
(select ssvid, scene_id, sum(probability) prob_sum from prob_multiplied_table
 group by ssvid, scene_id)
 using(ssvid,scene_id) )


 select ssvid, scene_id,
 sum(probability_norm) likelihood_in
 from normalized_prob
join
scene_footprints
using(scene_id)
where st_contains(st_geogfromtext(footprint), st_geogpoint(detect_lon, detect_lat))
group by ssvid, scene_id






# union all




# # select
# # ssvid,
# # scene_id,
# # seg_id,
# # detect_lat,
# # detect_lon,
# # old_lat,
# # old_lon,
# # lat_interpolate_adjusted,
# # lon_interpolate_adjusted,
# # label,
# # speed,
# # timestamp,
# # probability,
# # scale,
# # delta_minutes,
# #  from  messages_with_probabilities
# # where probability > 0




# select  * from scratch_david.interp_test"""


df = pd.read_gbq(q, project_id="world-fishing-827")


# -

df.head()

df.likelihood_in.sum()

len(df.ssvid.unique())


d = df.iloc[7]
d.delta_minutes1, d.delta_minutes2

d

d.lat_interpolate2

d.lon_interpolate1, d.lat_interpolate1

import math

d.lon_interpolate1 + math.sin(math.radians(d.course1)) * 0.1
d.lat_interpolate1 + math.cos(math.radians(d.course1)) * 0.1

d.course1, d.course2

plt.scatter(d.lon1, d.lat1, label="before")
plt.scatter(d.lon2, d.lat2, label="after")
plt.scatter(d.lon_interpolate1, d.lat_interpolate1, label="before interpolation")
plt.scatter(d.lon_interpolate2, d.lat_interpolate2, label="after interpolation")
plt.scatter(d.likely_lon, d.likely_lat, color="black", label="likely location")
# plt.scatter(d.lon_interpolate1 + math.sin(math.radians(d.course1))*1,
#             d.lat_interpolate1 + math.cos(math.radians(d.course1))*1, color = 'black')
plt.legend()
plt.title(
    "{},{},{}".format(round(d.delta_minutes1, 1), round(d.delta_minutes2, 1), d.ssvid)
)
plt.show()


for i in range(5):
    d = df.iloc[i]
    plt.scatter(d.lon1, d.lat1, label="before")
    plt.scatter(d.lon2, d.lat2, label="after")
    plt.scatter(d.lon_interpolate1, d.lat_interpolate1, label="before interpolation")
    plt.scatter(d.lon_interpolate2, d.lat_interpolate2, label="after interpolation")
    plt.scatter(d.likely_lon, d.likely_lat, color="black", label="likely location")
    # plt.scatter(d.lon_interpolate1 + math.sin(math.radians(d.course1))*1,
    #             d.lat_interpolate1 + math.cos(math.radians(d.course1))*1, color = 'black')
    plt.legend()
    plt.title(
        "{},{},{}".format(
            round(d.delta_minutes1, 1), round(d.delta_minutes2, 1), d.ssvid
        )
    )
    print(d.speed1, d.speed2, d.course1, d.course2)
    plt.show()


d

q = """ select
 ssvid,
 scene_id,
 a.detect_lat detect_lat,
 a.detect_lon detect_lon,
 delta_minutes,
 lat_interpolate_adjusted,
 lon_interpolate_adjusted,
 probability
 from scratch_david.walmart_rasters a
 where
 scene_id = 'RS2_20190831_150553_0074_DVWF_HH_SCS_753433_9526_29818035'
 and ssvid = '994010177' """
df = pd.read_gbq(q, project_id="world-fishing-827")


# +
q = f"""SELECT * FROM
`world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221`
where
--scene_id = '{scene_id}'
-- and
score > 0 and score < 1e-3"""

df_matched = pd.read_gbq(q, project_id="world-fishing-827")

# +
# scene_id = 'RS2_20191021_015301_0074_DVWF_HH_SCS_765981_0126_30985398'

# q = f''' select * from scratch_david.walmart_rasters
# where scene_id="{scene_id}" '''

# df= pd.read_gbq(q, project_id='world-fishing-827')
# -

q = """ select
 ssvid,
 scene_id,
 a.detect_lat detect_lat,
 a.detect_lon detect_lon,
 delta_minutes,
 lat_interpolate_adjusted,
 lon_interpolate_adjusted,
 probability
 from scratch_david.walmart_rasters a
 join
 proj_walmart_dark_targets.all_detections_and_ais_v20201221 b
 using(ssvid,scene_id)
 where score > 0 and score < 1e-3 """
df = pd.read_gbq(q, project_id="world-fishing-827")

# +
# df['detect_lon'] = df['detect_lat_1']

# +
q = f"""SELECT * FROM
`world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221`
where
--scene_id = '{scene_id}'
-- and
score > 0 and score < 1e-3"""

df_matched = pd.read_gbq(q, project_id="world-fishing-827")
# -

# !mkdir low_scores

# +
dotsize = 4
min_value = 1e-9


for index, row in df_matched.iterrows():

    ssvid = row.ssvid
    scene_id = row.scene_id
    score = row.score

    print(f"score: {score:.2E}")
    print(f"vessel {ssvid}")
    print(f"scene: {scene_id}")

    d = df[(df.ssvid == ssvid) & (df.scene_id == scene_id)]
    d_matched = df_matched[
        (df_matched.ssvid == ssvid) & (df_matched.scene_id == scene_id)
    ]
    num_lats = int((d.detect_lat.max() - d.detect_lat.min()) * 200)
    num_lons = int((d.detect_lon.max() - d.detect_lon.min()) * 200)

    grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

    min_lon = d.detect_lon.min()
    min_lat = d.detect_lat.min()
    max_lon = d.detect_lon.max()
    max_lat = d.detect_lat.max()

    scale = 200

    def fill_grid(r):
        y = int((r.detect_lat - min_lat) * scale)
        x = int((r.detect_lon - min_lon) * scale)
        grid[y][x] += r.probability

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

    grids = []
    lats_int = []
    lons_int = []
    minutes = d.delta_minutes.unique()

    for delta_minutes in minutes:

        d2 = d[d.delta_minutes == delta_minutes]
        grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

        def fill_grid(r):
            y = int((r.detect_lat - min_lat) * scale)
            x = int((r.detect_lon - min_lon) * scale)
            grid[y][x] += r.probability

        d2.apply(fill_grid, axis=1)

        grids.append(grid)
        lats_int.append(d2.lat_interpolate_adjusted.values[0])
        lons_int.append(d2.lon_interpolate_adjusted.values[0])

    norm = colors.LogNorm(vmin=min_value, vmax=grids[0].max())
    grids[0][grids[0] < min_value] = 0
    ax1.imshow(
        np.flipud(grids[0]), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat]
    )
    ax1.scatter(
        [lons_int[0]],
        [lats_int[0]],
        label="extrapolated position",
        s=dotsize,
        color="red",
    )
    ax1.scatter(
        d_matched.detect_lon.values,
        d_matched.detect_lat.values,
        color="black",
        label="matched detection",
        s=dotsize,
    )
    ax1.title.set_text(f"Minutes to scene: {round(minutes[0])}")

    if len(grids) > 1:
        norm = colors.LogNorm(vmin=min_value, vmax=grids[1].max())
        grids[1][grids[1] < min_value] = 0

        ax2.imshow(
            np.flipud(grids[1]), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat]
        )
        ax2.scatter(
            [lons_int[1]],
            [lats_int[1]],
            label="extrapolated position",
            s=dotsize,
            color="red",
        )
        ax2.scatter(
            d_matched.detect_lon.values,
            d_matched.detect_lat.values,
            color="black",
            label="matched detection",
            s=dotsize,
        )
        ax2.title.set_text(f"Minutes to scene: {round(minutes[1])}")

    if len(grids) > 1:
        for i in range(2):
            grids[i] = np.divide(grids[i], grids[i].sum())

        grid = np.multiply(grids[0], grids[1])
        grid = np.divide(grid, grid.sum())

    else:
        grid = np.divide(grids[0], grids[0].sum())

    grid[grid < min_value] = 0
    norm = colors.LogNorm(vmin=min_value, vmax=grid.max())
    ax3.imshow(np.flipud(grid), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat])
    ax3.scatter(
        d.lon_interpolate_adjusted.unique(),
        d.lat_interpolate_adjusted.unique(),
        label="extrapolated position",
        s=dotsize,
        color="red",
    )
    ax3.scatter(
        d_matched.detect_lon.values,
        d_matched.detect_lat.values,
        color="black",
        s=dotsize,
    )
    d = df_matched[df_matched.ssvid == ssvid]
    plt.scatter(
        d_matched.detect_lon.values,
        d_matched.detect_lat.values,
        color="black",
        label="matched detection",
        s=dotsize,
    )

    ax3.title.set_text("multiplied probability")
    plt.legend()
    fig.suptitle(f"{ssvid}, score: {score:0.2e}, scene: {scene_id}")
    plt.savefig(f"low_scores/{ssvid}_{scene_id}.png", bbox_inches="tight", dpi=300)
    plt.show()

# -

d_matched


# +
q = """with
# figure out where there is more than one seg_id in a scene
good_segs as (
--   select seg_id, scene_id from
--   (
--     select
--       scene_id, seg_id, ssvid, row_number() over (partition by seg_id, scene_id order by positions desc, min_delta_minutes) row_number
--     from
--     (
      select  seg_id, ssvid,scene_id, count(*) positions, min(abs(delta_minutes)) min_delta_minutes from (
      select distinct seg_id, ssvid, scene_id, delta_minutes from scratch_david.walmart_rasters)
      group by seg_id, ssvid, scene_id, scene_id
--     )
--   )
--   where row_number = 1
),

source_table as
(select * from scratch_david.walmart_rasters),

raster_sum1 as
(
  select
    scene_id, ssvid, detect_lat,
    detect_lon, seg_id, delta_minutes,
    probability/prob_sum as probability_norm
  from
    source_table
  join (
    select ssvid, scene_id, delta_minutes, sum(probability) prob_sum
    from source_table
    group by ssvid, scene_id, delta_minutes
  )
  using(ssvid, scene_id, delta_minutes)
),

raster_sum1_single as
(select
 scene_id, ssvid, detect_lat,
    detect_lon, seg_id,
    delta_minutes as delta_minutes_1,
    null as delta_minutes_2,
    probability_norm
from raster_sum1
join
good_segs
using(seg_id, ssvid,scene_id)
where positions  = 1
),


raster_sum1_double as
(select
 *
from raster_sum1
join
good_segs
using(seg_id, ssvid,scene_id)
where positions  = 2
),


multiplied_prob as (
  select
  a.scene_id scene_id,
  a.ssvid,
  a.seg_id,
  a.delta_minutes delta_minutes_1,
  b.delta_minutes delta_minutes_2,
  a.detect_lat,
  a.detect_lon,
  a.probability_norm*ifnull(b.probability_norm, 1) probability
  from
  raster_sum1_double a
  left join
  raster_sum1_double b
  on a.ssvid = b.ssvid and a.seg_id=b.seg_id
  and a.scene_id = b.scene_id
  and round(a.detect_lon*200) = round(b.detect_lon*200)
  and round(a.detect_lat*200) = round(b.detect_lat*200)
  where (a.delta_minutes > b.delta_minutes or b.delta_minutes is null)
),


multiplied_prob_norm as
(
  select
    scene_id, ssvid, detect_lat,
    detect_lon, seg_id,
    delta_minutes_1,
    delta_minutes_2,
    probability/prob_sum as probability_norm
  from
    multiplied_prob
  join (
    select ssvid, scene_id, delta_minutes_1, sum(probability) prob_sum
    from multiplied_prob
    group by ssvid, scene_id, delta_minutes_1
  )
  using(ssvid, scene_id, delta_minutes_1)
) ,

all_probabilities as (select * from
multiplied_prob_norm
union all
select * from raster_sum1_single),


footprints as (select distinct footprint, scene_id
 from proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 select distinct footprint, scene_id from
 proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110)

select scene_id, ssvid, sum(probability_norm) prob
from
all_probabilities
join
footprints
using(scene_id)
where st_contains( ST_GEOGFROMTEXT(footprint),st_geogpoint(detect_lon, detect_lat))
group by scene_id, ssvid


"""

df2 = pd.read_gbq(q, project_id="world-fishing-827")
# -

df.head()

# +


from random import random

xx = []
for i in range(10000):
    x = np.array(list(map(lambda x: int(float(x) > random()), df2.prob))).sum()
    xx.append(x)
# -

import seaborn as sns

sns.histplot(xx, bins=15)
plt.title("likely number of AIS vessels in scenes")

xx = np.array(xx)
xx.mean(), xx.std()
