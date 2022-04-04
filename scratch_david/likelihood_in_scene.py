# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
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

# +
# $2 query!

q = """#StandardSql
# Match AIS vessel detections to Sentinel-1 vessel detections


CREATE TEMP FUNCTION start_date() AS (DATE_SUB(DATE('2019-08-09'),INTERVAL 1 DAY ));
CREATE TEMP FUNCTION end_date() AS (DATE_ADD(DATE('2020-01-08'), INTERVAL 1 DAY));
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
  -- and ssvid = '81561798'
),
--
-- What is a good AIS segment? Find it here, in good_segs
good_segs as ( select seg_id from
           `world-fishing-827.gfw_research.pipe_v20201001_segs` where good_seg
),
--
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
        when best.best_vessel_class = "gear" then "gear"
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
ATAN2(E_Km,N_km)*180/3.1416 sat_course, -- convert to degrees from radians
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
  label,
  distance_from_shore_m,
  timestamp_diff(scene_timestamp, timestamp, SECOND) / 60.0
  as delta_minutes,
  scene_id,
  scene_timestamp,
  # the following two help a later join. Basically, we want to know if there is another image to join
  (timestamp_before is not null and abs(timestamp_diff(scene_timestamp, timestamp_before, SECOND)) / 60.0 < 9*60 ) previous_exists,
  (timestamp_after is not null and abs(timestamp_diff(scene_timestamp, timestamp_after, SECOND)) / 60.0 < 9*60 ) after_exists
from
  ais_messages_lagged_led a
join
  vessel_info_table b
on
  a.ssvid = b.ssvid
join
  scene_footprints
on
  abs(timestamp_diff(scene_timestamp, timestamp, SECOND)) / 60.0  < 9*60 # less than 9 hours
  and st_distance(st_geogpoint(lon, lat), st_geogfromtext(footprint)) < 100*1852 -- within 100 nautical miles of the scene
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
  where label != "gear"
and a.ssvid not in (SELECT ssvid FROM `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221` where gear)

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
  / 60  -- divided by the nautical miles per degree lat,
  -- which is 60 nautical miles per degree (really it is 59.9, and varies by lat)
  as lat_interpolate,
  reasonable_lon(
     lon + sin(radians(course)) -- y component
     *speed*delta_minutes/60 -- nautical miles (knots * minutes / (60 minutes / hour) )
     /(60*cos(radians(lat)))) -- divided by the nautical miles per degree lon,
  -- which is 60 times the cos of the lat (really it is 59.9, and varies by lat)
  as lon_interpolate,
  1000 / colmax(1.0, ABS(delta_minutes)) as scale,
  *
FROM
  best_messages),


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
  cos(radians(course))*500*1000/scale*meters2deg() as cosdegrees,
  sin(radians(course))*500*1000/scale*meters2deg() as sindegrees,
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
cosdegrees + sindegrees lat11, # actual order here might be wrong...
cosdegrees - sindegrees lat12,

# when farther from equator, cos(lat_interpolate) will be <1, so the
# actual distance in degrees lon will be bigger, so divide by cos(lat)
(cosdegrees + sindegrees) / cos(radians(lat_interpolate))  lon11, # actual order here might be wrong...
(cosdegrees - sindegrees) / cos(radians(lat_interpolate))  lon12,
 vessel_speed_perpendicular_to_sat / satellite_knots
 * pow( ( pow(vessel_distance_to_sat_nm,2) + pow(start_altitude,2)/pow(1852,2) ) , .5)
  -- divide by 1852 to convert meters to nautical miles,
  -- then use pathangerean theorm to get the approximate distance to the satellite in
  -- nautical miles.
  as adjusted_nautical_miles_parallel_to_sat
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
  * except(lon,lat),
  lat - adjusted_nautical_miles_parallel_to_sat * cos(radians(sat_course))/60 lat, -- 60 nautical miles per degree
  lon - adjusted_nautical_miles_parallel_to_sat * sin(radians(sat_course))/(60 * cos(radians(lat))) lon, -- 60 nautical miles * cos(lat) per degree
  lat_interpolate - adjusted_nautical_miles_parallel_to_sat * cos(radians(sat_course))/60 lat_interpolate_adjusted, -- 60 nautical miles per degree
  lon_interpolate - adjusted_nautical_miles_parallel_to_sat * sin(radians(sat_course))/(60 * cos(radians(lat))) lon_interpolate_adjusted, -- 60 nautical miles * cos(lat) per degree
  lat as old_lat,
  lon as old_lon
from
  interpolated_positions_adjust_formula
),




with_max_min_latlon as (

select
scene_id,
ssvid,
seg_id,
old_lat,
old_lon,
label,
 course, timestamp, scene_timestamp, lat_interpolate_adjusted,	lon_interpolate_adjusted, speed, delta_minutes, sindegrees, cosdegrees,
lat11,
lat12,
lon11,
lon12,
greatest(lat11,lat12,-lat11,-lat12) + lat_interpolate_adjusted max_lat,
least(lat11,lat12,-lat11,-lat12) + lat_interpolate_adjusted min_lat,
greatest(lon11,lon12,-lon11,-lon12) + lon_interpolate_adjusted max_lon,
least(lon11,lon12,-lon11,-lon12) + lon_interpolate_adjusted min_lon,
  scale
from best_messages_adjusted),



 lat_array AS(
  SELECT
    * except(lat),
    lat + .5/one_over_cellsize() as detect_lat  -- to get the middle of the cell
  FROM
    with_max_min_latlon
  CROSS JOIN
    UNNEST(GENERATE_ARRAY(FLOOR(min_lat*one_over_cellsize())/one_over_cellsize(),
    FLOOR(max_lat*one_over_cellsize())/one_over_cellsize(), 1/one_over_cellsize()))
     AS lat),


#     --
#     --
  lon_array AS (
  SELECT
    * except(lon),
    lon + .5/one_over_cellsize() as detect_lon -- to get the middle of the cell
  FROM
    with_max_min_latlon
  CROSS JOIN
    UNNEST(GENERATE_ARRAY(FLOOR(min_lon*one_over_cellsize())/one_over_cellsize(),
    FLOOR(max_lon*one_over_cellsize())/one_over_cellsize(), 1/one_over_cellsize())) AS lon),


  id_lat_lon_array AS (
  select
  detect_lat,
  detect_lon,
  a.old_lat,
  a.old_lon,
  a.label,
  a.speed,
  a.scene_id,
  a.ssvid,
  a.seg_id,
  a.course,
  a.timestamp,
  a.scene_timestamp,
  a.lat_interpolate_adjusted,
  a.lon_interpolate_adjusted,
  a.delta_minutes,
  a.sindegrees,
  a.cosdegrees,
  a.scale
  FROM
    lon_array a
  CROSS JOIN
    lat_array b
  WHERE
    a.scene_id=b.scene_id
    and a.ssvid=b.ssvid
    and a.timestamp = b.timestamp),


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




# --
# --
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
 )


select
ssvid,
scene_id,
seg_id,
detect_lat,
detect_lon,
old_lat,
old_lon,
lat_interpolate_adjusted,
lon_interpolate_adjusted,
label,
speed,
timestamp,
probability,
scale,
delta_minutes,
 from  messages_with_probabilities
where probability > 0
"""

# with open('temp.sql', 'w') as f:
#     f.write(q)

# command = '''cat temp.sql | bq query --replace \
#         --destination_table=scratch_david.walmart_rasters\
#          --allow_large_results --use_legacy_sql=false'''
# os.system(command)
# -

# !rm -f temp.sql


# +
scene_id = "RS2_20191021_015301_0074_DVWF_HH_SCS_765981_0126_30985398"

the_date = datetime.strptime(scene_id[4:15], "%Y%m%d_%H")
the_date
# -

h = the_date.hour
if h >= 12:
    start_date = the_date
    end_date = the_date + timedelta(days=1)
else:
    start_date = the_date - timedelta(days=1)
    end_date = the_date
start_date, end_date


# +
q = f"""#StandardSql
# Match AIS vessel detections to Sentinel-1 vessel detections

CREATE TEMP FUNCTION start_date() AS (DATE_SUB(DATE('{start_date:%Y-%m-%d}'),INTERVAL 1 DAY ));
CREATE TEMP FUNCTION end_date() AS (DATE_ADD(DATE('{end_date:%Y-%m-%d}'), INTERVAL 1 DAY));
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
where scene_id = '{scene_id}'
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
  -- and ssvid = '81561798'
),
--
-- What is a good AIS segment? Find it here, in good_segs
good_segs as ( select seg_id from
           `world-fishing-827.gfw_research.pipe_v20201001_segs` where good_seg
),
--
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
        when best.best_vessel_class = "gear" then "gear"
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
ATAN2(E_Km,N_km)*180/3.1416 sat_course, -- convert to degrees from radians
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
  label,
  distance_from_shore_m,
  timestamp_diff(scene_timestamp, timestamp, SECOND) / 60.0
  as delta_minutes,
  scene_id,
  scene_timestamp,
  # the following two help a later join. Basically, we want to know if there is another image to join
  (timestamp_before is not null and abs(timestamp_diff(scene_timestamp, timestamp_before, SECOND)) / 60.0 < 9*60 ) previous_exists,
  (timestamp_after is not null and abs(timestamp_diff(scene_timestamp, timestamp_after, SECOND)) / 60.0 < 9*60 ) after_exists
from
  ais_messages_lagged_led a
join
  vessel_info_table b
on
  a.ssvid = b.ssvid
join
  scene_footprints
on
  abs(timestamp_diff(scene_timestamp, timestamp, SECOND)) / 60.0  < 9*60 # less than 5 hours
  and st_distance(st_geogpoint(lon, lat), st_geogfromtext(footprint)) < 100*1852 -- within 100 nautical miles of the scene
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
where label != "gear"
and ssvid not in (SELECT ssvid FROM `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221` where gear)
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
  / 60  -- divided by the nautical miles per degree lat,
  -- which is 60 nautical miles per degree (really it is 59.9, and varies by lat)
  as lat_interpolate,
  reasonable_lon(
     lon + sin(radians(course)) -- y component
     *speed*delta_minutes/60 -- nautical miles (knots * minutes / (60 minutes / hour) )
     /(60*cos(radians(lat)))) -- divided by the nautical miles per degree lon,
  -- which is 60 times the cos of the lat (really it is 59.9, and varies by lat)
  as lon_interpolate,
  1000 / colmax(1.0, ABS(delta_minutes)) as scale,
  *
FROM
  best_messages),


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
  cos(radians(course))*500*1000/scale*meters2deg() as cosdegrees,
  sin(radians(course))*500*1000/scale*meters2deg() as sindegrees,
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
cosdegrees + sindegrees lat11, # actual order here might be wrong...
cosdegrees - sindegrees lat12,

# when farther from equator, cos(lat_interpolate) will be <1, so the
# actual distance in degrees lon will be bigger, so divide by cos(lat)
(cosdegrees + sindegrees) / cos(radians(lat_interpolate))  lon11, # actual order here might be wrong...
(cosdegrees - sindegrees) / cos(radians(lat_interpolate))  lon12,
 vessel_speed_perpendicular_to_sat / satellite_knots
 * pow( ( pow(vessel_distance_to_sat_nm,2) + pow(start_altitude,2)/pow(1852,2) ) , .5)
  -- divide by 1852 to convert meters to nautical miles,
  -- then use pathangerean theorm to get the approximate distance to the satellite in
  -- nautical miles.
  as adjusted_nautical_miles_parallel_to_sat
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
  * except(lon,lat),
  lat - adjusted_nautical_miles_parallel_to_sat * cos(radians(sat_course))/60 lat, -- 60 nautical miles per degree
  lon - adjusted_nautical_miles_parallel_to_sat * sin(radians(sat_course))/(60 * cos(radians(lat))) lon, -- 60 nautical miles * cos(lat) per degree
  lat_interpolate - adjusted_nautical_miles_parallel_to_sat * cos(radians(sat_course))/60 lat_interpolate_adjusted, -- 60 nautical miles per degree
  lon_interpolate - adjusted_nautical_miles_parallel_to_sat * sin(radians(sat_course))/(60 * cos(radians(lat))) lon_interpolate_adjusted, -- 60 nautical miles * cos(lat) per degree
  lat as old_lat,
  lon as old_lon
from
  interpolated_positions_adjust_formula
),




with_max_min_latlon as (

select
scene_id,
ssvid,
seg_id,
old_lat,
old_lon,
label,
 course, timestamp, scene_timestamp, lat_interpolate_adjusted,	lon_interpolate_adjusted, speed, delta_minutes, sindegrees, cosdegrees,
lat11,
lat12,
lon11,
lon12,
greatest(lat11,lat12,-lat11,-lat12) + lat_interpolate_adjusted max_lat,
least(lat11,lat12,-lat11,-lat12) + lat_interpolate_adjusted min_lat,
greatest(lon11,lon12,-lon11,-lon12) + lon_interpolate_adjusted max_lon,
least(lon11,lon12,-lon11,-lon12) + lon_interpolate_adjusted min_lon,
  scale
from best_messages_adjusted),



 lat_array AS(
  SELECT
    * except(lat),
    lat + .5/one_over_cellsize() as detect_lat  -- to get the middle of the cell
  FROM
    with_max_min_latlon
  CROSS JOIN
    UNNEST(GENERATE_ARRAY(FLOOR(min_lat*one_over_cellsize())/one_over_cellsize(),
    FLOOR(max_lat*one_over_cellsize())/one_over_cellsize(), 1/one_over_cellsize()))
     AS lat),


#     --
#     --
  lon_array AS (
  SELECT
    * except(lon),
    lon + .5/one_over_cellsize() as detect_lon -- to get the middle of the cell
  FROM
    with_max_min_latlon
  CROSS JOIN
    UNNEST(GENERATE_ARRAY(FLOOR(min_lon*one_over_cellsize())/one_over_cellsize(),
    FLOOR(max_lon*one_over_cellsize())/one_over_cellsize(), 1/one_over_cellsize())) AS lon),


  id_lat_lon_array AS (
  select
  detect_lat,
  detect_lon,
  a.old_lat,
  a.old_lon,
  a.label,
  a.speed,
  a.scene_id,
  a.ssvid,
  a.seg_id,
  a.course,
  a.timestamp,
  a.scene_timestamp,
  a.lat_interpolate_adjusted,
  a.lon_interpolate_adjusted,
  a.delta_minutes,
  a.sindegrees,
  a.cosdegrees,
  a.scale
  FROM
    lon_array a
  CROSS JOIN
    lat_array b
  WHERE
    a.scene_id=b.scene_id
    and a.ssvid=b.ssvid
    and a.timestamp = b.timestamp),


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




# --
# --
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
 )


select
ssvid,
seg_id,
detect_lat,
detect_lon,
old_lat,
old_lon,
lat_interpolate_adjusted,
lon_interpolate_adjusted,
label,
speed,timestamp,
probability,
scale,
delta_minutes,
 from  messages_with_probabilities
where probability > 0




"""

print(q)

# df = pd.read_gbq(q, project_id='world-fishing-827')
# -


df.head()

len(df.ssvid.unique())

df.ssvid.unique()

# +
# d = df[df.ssvid == '81561798']
# d
# -

df.columns

df.detect_lat.min(), df.detect_lat.max()

# +
q = f"""SELECT * FROM
`world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20201221`
where scene_id = '{scene_id}'
and score > 0"""

df_matched = pd.read_gbq(q, project_id="world-fishing-827")
# -

df_matched.scene_id.unique()

# +
# df.head()

# +
dotsize = 4


for ssvid in df_matched.ssvid.unique():

    d_matched = df_matched[df_matched.ssvid == ssvid]
    print("score:", d_matched.score.values[0])
    print(f"vessel {ssvid}")

    d = df[df.ssvid == ssvid]
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

    norm = colors.LogNorm(vmin=1e-9, vmax=grids[0].max())
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
        norm = colors.LogNorm(vmin=1e-9, vmax=grids[1].max())
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

    norm = colors.LogNorm(vmin=1e-9, vmax=grid.max())
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
    fig.suptitle(f"{ssvid} score: {d_matched.score.values[0]:0.2e}")
    plt.show()


# +

# scene RS2_20200107_144650_0074_DVWF_HH_SCS_785751_1158_32185414

dotsize = 4


for ssvid in df_matched.ssvid.unique():

    d_matched = df_matched[df_matched.ssvid == ssvid]
    print("score:", d_matched.score.values[0])
    print(f"vessel {ssvid}")

    d = df[df.ssvid == ssvid]
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

    norm = colors.LogNorm(vmin=1e-9, vmax=grids[0].max())
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
        norm = colors.LogNorm(vmin=1e-9, vmax=grids[1].max())
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

    norm = colors.LogNorm(vmin=1e-9, vmax=grid.max())
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
    fig.suptitle(f"{ssvid} score: {d_matched.score.values[0]:0.2e}")
    plt.show()


# +
dotsize = 4


for ssvid in df_matched.ssvid.unique():

    d_matched = df_matched[df_matched.ssvid == ssvid]
    print("score:", d_matched.score.values[0])
    print(f"vessel {ssvid}")

    d = df[df.ssvid == ssvid]
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

    norm = colors.LogNorm(vmin=1e-9, vmax=grids[0].max())
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
        norm = colors.LogNorm(vmin=1e-9, vmax=grids[1].max())
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

    norm = colors.LogNorm(vmin=1e-9, vmax=grid.max())
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
    fig.suptitle(f"{ssvid} score: {d_matched.score.values[0]:0.2e}")
    plt.show()

# -

print(f"{ssvid} score: {d_matched.score.values[0]:0.3e}")


plt.figure(figsize=(10, 10))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    num_lats = int((d.detect_lat.max() - d.detect_lat.min()) * 200)
    num_lons = int((d.detect_lon.max() - d.detect_lon.min()) * 200)

    min_lon = d.detect_lon.min()
    min_lat = d.detect_lat.min()
    max_lon = d.detect_lon.max()
    max_lat = d.detect_lat.max()

    scale = 200

    grids = []
    for delta_minutes in d.delta_minutes.unique():

        d2 = d[d.delta_minutes == delta_minutes]

        grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

        def fill_grid(r):
            y = int((r.detect_lat - min_lat) * scale)
            x = int((r.detect_lon - min_lon) * scale)
            grid[y][x] += r.probability

        d2.apply(fill_grid, axis=1)

        grids.append(grid)

    if len(grids) > 1:
        for i in range(2):
            grids[i] = np.divide(grids[i], grids[i].sum())

        grid = np.multiply(grids[0], grids[1])
        grid = np.divide(grid, grid.sum())
    else:
        grid = np.divide(grids[0], grids[0].sum())

    norm = colors.LogNorm(vmin=grid.max() / 1e6, vmax=grid.max())
    plt.imshow(np.flipud(grid), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat])
    plt.scatter(
        d.lon_interpolate_adjusted.unique(),
        d.lat_interpolate_adjusted.unique(),
        alpha=0.3,
    )
    # plt.title(ssvid)

    print(ssvid, d.delta_minutes.unique())
    plt.show()

# +
q = f"""
select * from (select distinct footprint, scene_id
 from proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117
 union all
 select distinct footprint, scene_id from
 proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110)
 where scene_id = '{scene_id}' """

df_f = pd.read_gbq(q, project_id="world-fishing-827")
import shapely.wkt

polygon = shapely.wkt.loads(df_f.footprint.values[0])

# +
plt.figure(figsize=(10, 10))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    num_lats = int((d.detect_lat.max() - d.detect_lat.min()) * 200)
    num_lons = int((d.detect_lon.max() - d.detect_lon.min()) * 200)

    min_lon = d.detect_lon.min()
    min_lat = d.detect_lat.min()
    max_lon = d.detect_lon.max()
    max_lat = d.detect_lat.max()

    scale = 200

    grids = []
    for delta_minutes in d.delta_minutes.unique():

        d2 = d[d.delta_minutes == delta_minutes]

        grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

        def fill_grid(r):
            y = int((r.detect_lat - min_lat) * scale)
            x = int((r.detect_lon - min_lon) * scale)
            grid[y][x] += r.probability

        d2.apply(fill_grid, axis=1)

        grids.append(grid)

    if len(grids) > 1:
        for i in range(2):
            grids[i] = np.divide(grids[i], grids[i].sum())

        grid = np.multiply(grids[0], grids[1])
        grid = np.divide(grid, grid.sum())
    else:
        grid = np.divide(grids[0], grids[0].sum())

    norm = colors.LogNorm(vmin=grid.max() / 1e6, vmax=grid.max())
    plt.imshow(np.flipud(grid), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat])
    plt.scatter(
        d.lon_interpolate_adjusted.unique(),
        d.lat_interpolate_adjusted.unique(),
        s=2,
        color="black",
    )
    # plt.title(ssvid)

    print(ssvid, d.delta_minutes.unique())
polygon = shapely.wkt.loads(df_f.footprint.values[0])

x, y = polygon.exterior.xy
plt.plot(x, y)
plt.scatter(
    df_matched.detect_lon.values,
    df_matched.detect_lat.values,
    color="red",
    s=2,
    label="SAR detects",
)
plt.legend()
plt.show()


# +
plt.figure(figsize=(10, 10))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    num_lats = int((d.detect_lat.max() - d.detect_lat.min()) * 200)
    num_lons = int((d.detect_lon.max() - d.detect_lon.min()) * 200)

    min_lon = d.detect_lon.min()
    min_lat = d.detect_lat.min()
    max_lon = d.detect_lon.max()
    max_lat = d.detect_lat.max()

    scale = 200

    grids = []
    for delta_minutes in d.delta_minutes.unique():

        d2 = d[d.delta_minutes == delta_minutes]

        grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

        def fill_grid(r):
            y = int((r.detect_lat - min_lat) * scale)
            x = int((r.detect_lon - min_lon) * scale)
            grid[y][x] += r.probability

        d2.apply(fill_grid, axis=1)

        grids.append(grid)

    if len(grids) > 1:
        for i in range(2):
            grids[i] = np.divide(grids[i], grids[i].sum())

        grid = np.multiply(grids[0], grids[1])
        grid = np.divide(grid, grid.sum())
    else:
        grid = np.divide(grids[0], grids[0].sum())

    norm = colors.LogNorm(vmin=grid.max() / 1e6, vmax=grid.max())
    plt.imshow(np.flipud(grid), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat])
    plt.scatter(
        d.lon_interpolate_adjusted.unique(),
        d.lat_interpolate_adjusted.unique(),
        s=2,
        color="black",
    )
    # plt.title(ssvid)

    print(ssvid, d.delta_minutes.unique())
polygon = shapely.wkt.loads(df_f.footprint.values[0])

x, y = polygon.exterior.xy
plt.plot(x, y)
plt.scatter(
    df_matched.detect_lon.values,
    df_matched.detect_lat.values,
    color="red",
    s=2,
    label="SAR detects",
)
plt.legend()
plt.show()


# -


# +
polygon = shapely.wkt.loads(df_f.footprint.values[0])

odds_inside_list = []

plt.figure(figsize=(10, 10))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    num_lats = int((d.detect_lat.max() - d.detect_lat.min()) * 200)
    num_lons = int((d.detect_lon.max() - d.detect_lon.min()) * 200)

    if num_lats == 1 or num_lons == 1:
        continue

    min_lon = d.detect_lon.min()
    min_lat = d.detect_lat.min()
    max_lon = d.detect_lon.max()
    max_lat = d.detect_lat.max()

    grid_inside = np.zeros(shape=(num_lats + 1, num_lons + 1))
    lons = np.arange(min_lon + 0.5 / 200, max_lon + 0.5 / 200 + 200, 1 / 200)
    lats = np.arange(min_lat + 0.5 / 200, max_lat + 0.5 / 200 + 200, 1 / 200)

    for i in range(num_lats + 1):
        for j in range(num_lons + 1):
            grid_inside[i][j] = polygon.contains(Point(lons[j], lats[i]))

    scale = 200

    grids = []
    for delta_minutes in d.delta_minutes.unique():

        d2 = d[d.delta_minutes == delta_minutes]

        grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

        def fill_grid(r):
            y = int((r.detect_lat - min_lat) * scale)
            x = int((r.detect_lon - min_lon) * scale)
            grid[y][x] += r.probability

        d2.apply(fill_grid, axis=1)

        grids.append(grid)

    if len(grids) > 1:
        for i in range(2):
            grids[i] = np.divide(grids[i], grids[i].sum())

        grid = np.multiply(grids[0], grids[1])
        grid = np.divide(grid, grid.sum())
    else:
        grid = np.divide(grids[0], grids[0].sum())

    odds_inside = np.multiply(grid_inside, grid).sum()
    print(odds_inside)
    odds_inside_list.append([ssvid, odds_inside])

    if odds_inside > 0 and odds_inside < 0.99:

        norm = colors.LogNorm(vmin=grid.max() / 1e6, vmax=grid.max())
        plt.imshow(
            np.flipud(grid), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat]
        )
        plt.scatter(
            d.lon_interpolate_adjusted.unique(),
            d.lat_interpolate_adjusted.unique(),
            s=2,
            color="black",
        )

    print(ssvid, d.delta_minutes.unique())

x, y = polygon.exterior.xy
plt.plot(x, y)
plt.scatter(
    df_matched.detect_lon.values,
    df_matched.detect_lat.values,
    color="red",
    s=2,
    label="SAR detects",
)
plt.legend()
plt.show()


# +
polygon = shapely.wkt.loads(df_f.footprint.values[0])

odds_inside_list = []

plt.figure(figsize=(10, 10))
for ssvid in df.ssvid.unique():
    d = df[df.ssvid == ssvid]
    num_lats = int((d.detect_lat.max() - d.detect_lat.min()) * 200)
    num_lons = int((d.detect_lon.max() - d.detect_lon.min()) * 200)

    if num_lats == 1 or num_lons == 1:
        continue

    min_lon = d.detect_lon.min()
    min_lat = d.detect_lat.min()
    max_lon = d.detect_lon.max()
    max_lat = d.detect_lat.max()

    grid_inside = np.zeros(shape=(num_lats + 1, num_lons + 1))
    lons = np.arange(min_lon + 0.5 / 200, max_lon + 0.5 / 200 + 200, 1 / 200)
    lats = np.arange(min_lat + 0.5 / 200, max_lat + 0.5 / 200 + 200, 1 / 200)

    for i in range(num_lats + 1):
        for j in range(num_lons + 1):
            grid_inside[i][j] = polygon.contains(Point(lons[j], lats[i]))

    scale = 200

    grids = []
    for delta_minutes in d.delta_minutes.unique():

        d2 = d[d.delta_minutes == delta_minutes]

        grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

        def fill_grid(r):
            y = int((r.detect_lat - min_lat) * scale)
            x = int((r.detect_lon - min_lon) * scale)
            grid[y][x] += r.probability

        d2.apply(fill_grid, axis=1)

        grids.append(grid)

    if len(grids) > 1:
        for i in range(2):
            grids[i] = np.divide(grids[i], grids[i].sum())

        grid = np.multiply(grids[0], grids[1])
        grid = np.divide(grid, grid.sum())
    else:
        grid = np.divide(grids[0], grids[0].sum())

    odds_inside = np.multiply(grid_inside, grid).sum()
    print(odds_inside)
    odds_inside_list.append([ssvid, odds_inside])

    if odds_inside > 0 and odds_inside < 0.99:

        norm = colors.LogNorm(vmin=grid.max() / 1e6, vmax=grid.max())
        plt.imshow(
            np.flipud(grid), norm=norm, extent=[min_lon, max_lon, min_lat, max_lat]
        )
        plt.scatter(
            d.lon_interpolate_adjusted.unique(),
            d.lat_interpolate_adjusted.unique(),
            s=2,
            color="black",
        )

    print(ssvid, d.delta_minutes.unique())

x, y = polygon.exterior.xy
plt.plot(x, y)
plt.scatter(
    df_matched.detect_lon.values,
    df_matched.detect_lat.values,
    color="red",
    s=2,
    label="SAR detects",
)
plt.legend()
plt.show()


# -

from tabulate import tabulate

print(tabulate(odds_inside_list))

# +
a = np.array(odds_inside_list)[:, 1]


# -

b = list(map(float, a))
b.sort()
b

from random import random

xx = []
for i in range(10000):
    x = np.array(list(map(lambda x: int(float(x) > random()), a))).sum()
    xx.append(x)

import seaborn as sns

sns.distplot(xx)

np.array(xx).mean()

np.arange(min_lon + 0.5 / 200, max_lon + 0.5 / 200, 1)

max_lon, min_lon

d.detect_lon.max(), d.detect_lon.min()

df_f.footprint.values[0]

# +
grid = np.zeros(shape=(num_lats + 1, num_lons + 1))


def fill_grid(r):
    y = int((r.detect_lat - min_lat) * scale)
    x = int((r.detect_lon - min_lon) * scale)
    grid[y][x] += r.probability


d2.apply(fill_grid, axis=1)

grids.append(grid)
# -

num_lats


d2


num_lats = int((df.detect_lat.max() - df.detect_lat.min()) * 200)
num_lons = int((df.detect_lon.max() - df.detect_lon.min()) * 200)

num_lats, num_lons

len(df.detect_lat.unique()), len(df.detect_lon.unique())

# +
grid = np.zeros(shape=(num_lats + 1, num_lons + 1))

min_lon = df.detect_lon.min()
min_lat = df.detect_lat.min()
max_lon = df.detect_lon.max()
max_lat = df.detect_lat.max()

scale = 200


def fill_grid(r):
    y = int((r.detect_lat - min_lat) * scale)
    x = int((r.detect_lon - min_lon) * scale)
    grid[y][x] = r.probability


df.apply(fill_grid, axis=1)

None


# -

import math

import cartopy
import cartopy.crs as ccrs
import cmocean
import geopandas as gpd
# # %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# import pyseas.rasters
import pandas as pd
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

norm = colors.LogNorm(vmin=1e-7, vmax=0.08)
plt.imshow(grid, norm=norm, extent=[min_lon, max_lon, min_lat, max_lat])
plt.scatter(
    [df.lon_interpolate_adjusted.values[0]], [df.lat_interpolate_adjusted.values[0]]
)

df.columns

df.course_rads.unique()
