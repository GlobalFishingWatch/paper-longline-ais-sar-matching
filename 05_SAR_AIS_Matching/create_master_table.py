# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Create table with all vessels and detections
#

# %%
import pandas as pd

# %%
# inputs
matched_table = "proj_walmart_dark_targets.matching_v20210421_4_matched_combined"
detections_fp = "proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117"
detections_ind = "proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110"
extrapolated_table = "proj_walmart_dark_targets.matching_v20210421_1_extrapolated_ais"
Vinfo_table = "world-fishing-827.gfw_research.vi_ssvid_v20210301"
vessel_class = "proj_walmart_dark_targets.all_mmsi_vessel_class"
longliner_binned_hours = "proj_walmart_dark_targets.longline_fishing_hours_v20210318"


# when regions are unioned
# matched_union = 'scratch_pete.multiplied_test_v20210409_bothRegions_3_matched'

output_table = "proj_walmart_dark_targets.all_detections_and_ais_v20210427"

# %%
q = f"""
with

AOI as (
SELECT distinct 'indian' as region,  footprint as footprint from  `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110`
union all
select distinct 'pacific' as region,  footprint as footprint from `world-fishing-827.proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117`
),

footprints as (
select region, ST_UNION_AGG(ST_GEOGFROMTEXT(footprint)) footprint
from AOI
group by region),

all_matched as (
  select if(detect_lon < 0, "pacific", "indian") as region, *
  from  `world-fishing-827.{matched_table}`
  --select *,
  --if(ST_intersects(ST_GEOGPOINT(a.detect_lon, a.detect_lat), (select footprint from footprints where region = 'indian')), 'indian', 'pacific') as region
  --from `scratch_pete.multiplied_test_v20210409_bothRegions_3_matched`a
),
#
sar_lengths as (
  select Length AS sar_length, DetectionId as detect_id from
    (select * from `proj_walmart_dark_targets.walmart_ksat_detections_fp_v20200117`
     union all
    select * from `proj_walmart_dark_targets.walmart_ksat_detections_ind_v20200110`
    )
),
#
likely_locations as (
select if(likely_lon < 0, "pacific", "indian") as region,
* from `{extrapolated_table}`
),

joined_tables as (
select
  region,
  scene_id,
  detect_id,
  ssvid,
  detect_lat,
  detect_lon,
  likely_lat as lat_best,
  likely_lon as lon_best,
  score,
  best.best_length_m as gfw_length,
  sar_length,
  within_footprint,
  within_footprint_5km,
  within_footprint_1km,
  if (final_vessel_class = 'gear', True, false) as gear,
  final_vessel_class as vessel_type
from
  all_matched
full outer join
  likely_locations
using(ssvid, scene_id, region)
left join
  `world-fishing-827.gfw_research.vi_ssvid_v20210301`
using(ssvid)
left join
  proj_walmart_dark_targets.all_mmsi_vessel_class
using(ssvid)
left join
  sar_lengths
using(detect_id)
),

final_table as (
select *
from joined_tables
left join
`proj_walmart_dark_targets.longline_fishing_hours_v20210318`
on lat_bin = floor(ifnull(detect_lat, lat_best)*4)
and lon_bin = floor(ifnull(detect_lon, lon_best)*4))


select * from final_table
left join
(select
   match_category as match_review, notes, ssvid,
   detect_id
from proj_walmart_dark_targets.detections_reviewed_v20210427 )
using(detect_id, ssvid)



"""

# print(q)

df = pd.read_gbq(q, project_id="world-fishing-827")

# %%
df.head()

# %%
df.to_gbq(output_table, project_id="world-fishing-827", if_exists="replace")

# %%
# add territories to the table
q = f"""with
added_fields as (
select
*,
if (vessel_type in ("drifting_longlines",
                "drifting_longline",
                "tuna_purse_seines",
                "fishing"), true, false) is_fishing,
from {output_table}
left join
`world-fishing-827.proj_walmart_dark_targets.likelihood_in_scene`
using(scene_id, ssvid)
left join
(select ssvid, best.best_flag from `world-fishing-827.gfw_research.vi_ssvid_v20210301` )
using(ssvid)
),
with_eez as (
select scene_id, ssvid, detect_id,ISO_TER1, TERRITORY1 from
{output_table}
  CROSS JOIN
  (select wkt, MRGID, ISO_TER1, TERRITORY1 from `world-fishing-827.minderoo.marine_regions_v11`)
  WHERE
  ST_CONTAINS(SAFE.ST_GEOGFROMTEXT(wkt),ST_GEOGPOINT(ifnull(detect_lon, lon_best),ifnull(detect_lat, lat_best))))

select * except(ISO_TER1, TERRITORY1),
 ifnull(ISO_TER1, "high seas") ISO3,
 ifnull(TERRITORY1,"high seas") territory
 from added_fields
 left join
 with_eez
 using(scene_id, ssvid, detect_id)"""

df2 = pd.read_gbq(q, project_id="world-fishing-827")

# %%
df2.head()

# %%
df2.to_gbq(output_table, project_id="world-fishing-827", if_exists="replace")

# %%

# %%
