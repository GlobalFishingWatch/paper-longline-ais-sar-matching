# %%
import pandas as pd
import pyseas
import pyseas.styles
import matplotlib.pyplot as plt
from pyseas.contrib import plot_tracks
import pyseas.maps as psm
import pyseas.styles
from collections import Counter
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as tkr

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

#colors
purple = '#d73b68'
navy = '#204280'
orange = '#f68d4b'
gold = '#f8ba47'
green = '#ebe55d'

def gbq(q):
    return pd.read_gbq(q, project_id = 'world-fishing-827')


# %% [markdown]
# ## IOTC + Taiwan registry matches. 

# %%
iotc = '''



-- For each fleet in the Indian ocean 
-- I'd like to know how many longline vessels are on the IOTC registry, 
-- and how many we have matched to AIS.
WITH
vdb AS (
SELECT *
FROM `world-fishing-827.vessel_database.all_vessels_v20210401`
),

---------------------------------------------------
-- IOTC registered vessels matched to AIS
-- Taiwanese vessels are added separately
-- with registry codes TWN and TWN2
-- the former for >24m the latter <24m
-- all assumed to be drifting longliners
-- Matched SSVID can be unique identifiable numbers
---------------------------------------------------
iotc_matched AS (
SELECT DISTINCT
CONCAT (
  udfs.extract_regcode (r.list_uvi),
  SUBSTR (
    r.list_uvi, 
    LENGTH (SPLIT (r.list_uvi, "-")[OFFSET(0)]) + 1, 
    LENGTH (r.list_uvi))) AS list_uvi, 
r.shipname, 
r.flag,
"IOTC" AS registry,
Matched, 
FROM vdb
LEFT JOIN UNNEST (registry) AS r
WHERE 
  (r.list_uvi LIKE "IOTC%"
  OR r.list_uvi LIKE "TWN-%"
  OR r.list_uvi LIKE "TWN2-%" )
  AND matched
  AND list_uvi NOT IN (
    SELECT list_uvi
    FROM vdb
    LEFT JOIN UNNEST (registry)
    WHERE not matched
      AND list_uvi IS NOT NULL )
  AND authorized_from <= TIMESTAMP("2019-12-31")
  AND authorized_to >= TIMESTAMP("2019-01-01")
  AND flag in ('CHN', 'KOR', 'SYC', 'TWN' )
  AND ( (list_uvi LIKE "IOTC%" AND geartype LIKE "%drifting_longlines%")
    OR list_uvi NOT LIKE "IOTC%" )
),
  
------------------------------------------------
-- IOTC registered vessels unmatched to AIS
-- Taiwanese vessels are added separately
-- same as above
-- Remove those of very old records by filtering
-- authorization ranges before 2012
-- Unique identifiable numbers are list_uvi
------------------------------------------------

iotc_unmatched AS (
SELECT DISTINCT 
CONCAT (
  udfs.extract_regcode (r.list_uvi),
  SUBSTR (
    r.list_uvi, 
    LENGTH (SPLIT (r.list_uvi, "-")[OFFSET(0)]) + 1, 
    LENGTH (r.list_uvi))) AS list_uvi, 
r.shipname, 
r.flag,
"IOTC" AS registry,
False as Matched
FROM vdb
LEFT JOIN UNNEST (registry) AS r
WHERE 
  (r.list_uvi LIKE "IOTC%"
  OR r.list_uvi LIKE "TWN-%"
  OR r.list_uvi LIKE "TWN2-%" )
  AND NOT matched
  AND list_uvi NOT IN (
    SELECT list_uvi
    FROM vdb
    LEFT JOIN UNNEST (registry)
    WHERE matched
      AND list_uvi IS NOT NULL )
  AND authorized_from <= TIMESTAMP("2019-12-31")
  AND authorized_to >= TIMESTAMP("2019-01-01")
  AND flag in ('CHN', 'KOR', 'SYC', 'TWN' )
  AND ( (list_uvi LIKE "IOTC%" AND geartype LIKE "%drifting_longlines%")
    OR list_uvi NOT LIKE "IOTC%" )),


iotc_longliners as (
select * from iotc_matched
union distinct
select * from iotc_unmatched
),



matched_count as (
select
flag,
COUNT (DISTINCT list_uvi) AS Matched
FROM iotc_longliners
where matched
GROUP BY flag),

not_matched_count as (
select
flag,
 COUNT (DISTINCT list_uvi) AS not_matched
FROM iotc_longliners
where not matched
GROUP BY flag
)
select distinct flag as Flag, Case
  when flag = 'TWN' THEN 'Fishing entity of Taiwan'
  when flag = 'KOR' THEN 'Republic of Korea'
  else territory1
  end
  as Country,
  matched as Matched, not_matched, 
From matched_count
full outer join
not_matched_count
using(flag)
left join `world-fishing-827.gfw_research.eez_info`
on (flag = territory1_iso3)
order by flag

'''

# %%
iotc_df = gbq(iotc)

# %%
iotc_df = iotc_df.fillna(0)

# %%
iotc_df['Total Count'] = iotc_df.Matched + iotc_df.not_matched

# %%
iotc_df = iotc_df.sort_values('Country', ascending = False)
iotc_df = iotc_df.rename(columns={'not_matched': 'Not matched'})
iotc_df

# %% [markdown]
# ## Repeat for French Polynesia using WCPFC

# %%
wcpfc = '''
WITH
vdb AS (
  SELECT *
  FROM `world-fishing-827.vessel_database.all_vessels_v20210401`
),

------------------------------------------------
-- wcpfc registered vessels matched to AIS
------------------------------------------------
wcpfc_matched AS (
  SELECT DISTINCT
  CONCAT (
    udfs.extract_regcode (r.list_uvi),
    SUBSTR (
      r.list_uvi, 
      LENGTH (SPLIT (r.list_uvi, "-")[OFFSET(0)]) + 1, 
      LENGTH (r.list_uvi))) AS list_uvi, 
  r.shipname, 
  r.flag,
  "IOTC" AS registry,
  Matched, 
  FROM vdb
  LEFT JOIN UNNEST (registry) AS r
      WHERE 
  r.list_uvi LIKE "%WCPFC%"
  AND matched
  AND list_uvi NOT IN (
    SELECT list_uvi
    FROM vdb
    LEFT JOIN UNNEST (registry)
    WHERE not matched
      AND list_uvi IS NOT NULL )
  AND authorized_from <= TIMESTAMP("2019-12-31")
  AND authorized_to >= TIMESTAMP("2019-01-01") 
  AND flag in ('CHN', 'KOR', 'PYF','TWN', 'VUT' )
  AND geartype LIKE "%drifting_longlines%"
),

------------------------------------------------
-- wcpfc registered vessels unmatched to AIS
------------------------------------------------
wcpfc_unmatched AS (
SELECT DISTINCT 
  CONCAT (
    udfs.extract_regcode (r.list_uvi),
    SUBSTR (
      r.list_uvi, 
      LENGTH (SPLIT (r.list_uvi, "-")[OFFSET(0)]) + 1, 
      LENGTH (r.list_uvi))) AS list_uvi, 
  r.shipname, 
  r.flag,
  "WCPFC" AS registry,
  False as matched
FROM vdb
LEFT JOIN UNNEST (registry) AS r
WHERE 
    r.list_uvi LIKE "%WCPFC%"
    AND NOT matched
    AND list_uvi NOT IN (
      SELECT list_uvi
      FROM vdb
      LEFT JOIN UNNEST (registry)
      WHERE matched
        AND list_uvi IS NOT NULL )
    AND authorized_from <= TIMESTAMP("2019-12-31")
    AND authorized_to >= TIMESTAMP("2019-01-01") 
    AND flag in ('CHN', 'KOR', 'PYF','TWN', 'VUT' )
    AND geartype LIKE "%drifting_longlines%"),


wcpfc_longliners as (
select * from wcpfc_matched
UNION DISTINCT
select * from wcpfc_unmatched),
  
matched_count as (
select
flag,
COUNT (DISTINCT list_uvi) AS matched
FROM wcpfc_longliners
where matched
GROUP BY flag),

not_matched_count as (
select
flag,
 COUNT (DISTINCT list_uvi) AS not_matched
FROM wcpfc_longliners
where not matched
GROUP BY flag
)

select distinct flag as Flag, 
Case
  when flag = 'TWN' THEN "Fishing entity of Taiwan"
  when flag = 'KOR' THEN 'Republic of Korea'
  else territory1
  end
  as Country,
matched as Matched,
not_matched, 
From matched_count
full outer join
not_matched_count
using(flag)
left join `world-fishing-827.gfw_research.eez_info`
on (flag = territory1_iso3)
order by flag

'''

# %%
wcpfc = gbq(wcpfc)

# %%
wcpfc = wcpfc.fillna(0)
wcpfc= wcpfc.sort_values('Country', ascending = False)
wcpfc = wcpfc.rename(columns={'not_matched': 'Not matched'})
wcpfc['Total Count'] = wcpfc.Matched + wcpfc['Not matched']
wcpfc

# %% [markdown]
# ## Repeat for French Polynesia using IATTC

# %%
iattc = '''
WITH

vdb AS (
SELECT *
FROM `world-fishing-827.vessel_database.all_vessels_v20210401`
),

------------------------------------------------
-- iattc registered vessels matched to AIS
------------------------------------------------
iattc_matched AS (
SELECT DISTINCT 
CONCAT (
  udfs.extract_regcode (r.list_uvi),
  SUBSTR (
    r.list_uvi, 
    LENGTH (SPLIT (r.list_uvi, "-")[OFFSET(0)]) + 1, 
    LENGTH (r.list_uvi))) AS list_uvi, 
r.shipname, 
r.flag,
"IATTC" AS registry,
Matched
FROM vdb
LEFT JOIN UNNEST (registry) AS r
WHERE r.list_uvi LIKE "%IATTC%"
  AND matched
AND list_uvi NOT IN (
  SELECT list_uvi
  FROM vdb
  LEFT JOIN UNNEST (registry)
  WHERE not matched
    AND list_uvi IS NOT NULL )
  AND scraped BETWEEN "2019-01-01" AND "2019-12-31"
  AND flag in ('CHN', 'KOR', 'PYF','TWN', 'VUT' )
  AND geartype LIKE "%drifting_longlines%"
),
  
------------------------------------------------
-- iattc registered vessels unmatched to AIS
------------------------------------------------
iattc_unmatched AS (
SELECT DISTINCT 
CONCAT (
  udfs.extract_regcode (r.list_uvi),
  SUBSTR (
    r.list_uvi, 
    LENGTH (SPLIT (r.list_uvi, "-")[OFFSET(0)]) + 1, 
    LENGTH (r.list_uvi))) AS list_uvi, 
r.shipname, 
r.flag,
"IATTC" AS registry,
False as Matched
FROM vdb
LEFT JOIN UNNEST (registry) AS r
WHERE r.list_uvi LIKE "%IATTC%"
  AND NOT matched
AND list_uvi NOT IN (
  SELECT list_uvi
  FROM vdb
  LEFT JOIN UNNEST (registry)
  WHERE matched
    AND list_uvi IS NOT NULL )
  AND scraped BETWEEN "2019-01-01" AND "2019-12-31"
  AND flag in ('CHN', 'KOR', 'PYF','TWN', 'VUT' )
  AND geartype LIKE "%drifting_longlines%"
),
 

iattc_longliners as (
select *
from iattc_matched

UNION DISTINCT

SELECT *
FROM iattc_unmatched
),
  
  
matched_count as (
select
flag,
COUNT (DISTINCT list_uvi) AS matched
FROM iattc_longliners
where matched
GROUP BY flag),

not_matched_count as (
select
flag,
 COUNT (DISTINCT list_uvi) AS not_matched
FROM iattc_longliners
where not matched
GROUP BY flag
)

select distinct flag as Flag, 
Case
  when flag = 'TWN' THEN "Fishing entity of Taiwan"
  when flag = 'KOR' THEN 'Republic of Korea'
  else territory1
  end
  as Country, 
  matched as Matched, 
  not_matched, 
From matched_count
full outer join
not_matched_count
using(flag)
left join `world-fishing-827.gfw_research.eez_info`
on (flag = territory1_iso3)
order by flag
'''

# %%
iattc = gbq(iattc)

# %%
iattc = iattc.fillna(0)
iattc = iattc.sort_values('Country', ascending = False)
iattc = iattc.rename(columns={'not_matched': 'Not matched'})
iattc

# %%
iattc['total_count'] = iattc.Matched + iattc['Not matched']
df_total = iattc['total_count']
iattc_plt = iattc.iloc[:,1:4]
iattc_plt

# %% [markdown]
# ### Figure 4

# %%
q = '''select distinct vessel_count ,
region,
ifnull(flag_state, 'None') as flag_state,
Case
  when flag_state = 'TWN' THEN "Fishing entity of Taiwan"
  when flag_state = 'KOR' THEN 'Republic of Korea'
  else territory1
  end
  as Country,
Case
  when territory = 'Republic of Mauritius' THEN "Republic of Mauritius EEZ"
  when territory = 'Seychelles' THEN 'Seychelles EEZ'
  when territory = 'Madagascar' THEN 'Madagascar EEZ'
  when territory = 'high seas' THEN 'High seas'
  else territory
  end
  as territory,
from (
SELECT count (ssvid) as vessel_count ,
territory,
region,
best_flag as flag_state
FROM `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20210427`
where vessel_type = 'drifting_longlines'
and likelihood_in >0.5
group by territory, region, flag_state)
left join `world-fishing-827.gfw_research.eez_info`
on (flag_state = territory1_iso3)
order by region, vessel_count desc
'''
eez_count = gbq(q)

# %%
eez_count.head(50)

# %%
fp_eez_count = eez_count.loc[eez_count["region"] == "pacific"].set_index('Country')
mad_eez_count = eez_count.loc[eez_count["region"] == "indian"].set_index('Country')

mad_eez_count2 = mad_eez_count.pivot_table(values='vessel_count', index=mad_eez_count.index, columns='territory', aggfunc='first').sort_index()
fp_eez_count2 = fp_eez_count.pivot_table(values='vessel_count', index=fp_eez_count.index, columns='territory', aggfunc='first').sort_index()

# %%
iattc = iattc.loc[iattc['Flag'] != 'PYF']
wcpfc = wcpfc.loc[wcpfc['Flag'] != 'PYF']

# %%
# horizontal plotting
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['legend.title_fontsize'] = 24
fig4 = plt.figure(figsize=(25, 15))
gs = fig4.add_gridspec(ncols=6, nrows=2)
f4_ax1 = fig4.add_subplot(gs[1, 2:4])
f4_ax2 = fig4.add_subplot(gs[1, 4:6])
f4_ax3 = fig4.add_subplot(gs[1, :2])
f4_ax4 = fig4.add_subplot(gs[0, 2:4])
f4_ax5 = fig4.add_subplot(gs[0, :2])


df_total = iotc_df['Total Count'].astype(int)

iotc_order = ['Republic of Korea', 'China', 'Seychelles', \
              'Fishing entity of Taiwan']
iotc_df.Country = iotc_df.Country.astype("category")
iotc_df.Country.cat.set_categories(iotc_order, inplace=True)
iotc_df = iotc_df.sort_values('Country')

iotc_plt = iotc_df.iloc[:,1:4]

p1 = iotc_plt.plot(x = 'Country', kind='barh', ax = f4_ax4, width=0.8,
                   stacked = True,
              mark_right = True, color = ['mediumseagreen', 'lightslategray'], 
                   xlabel = '', ylabel = '', legend = False, fontsize = 20)
p1.set_title(r"$\bf{" + 'b' + "}$" + ' Indian Ocean Tuna\nCommission', 
             fontsize = 30)
# p1.set_ylabel('Flag State', fontsize = 18)
l = p1.legend(title='AIS vessels\nmatched to registries', loc='lower right', 
              frameon=False, fontsize = 22)
plt.setp(l.get_title(), multialignment='center')
p1.grid(False)
f4_ax4.set_yticklabels([])
f4_ax4.tick_params(axis='both', which='major', labelsize=22)


df_rel3 = iotc_plt[iotc_plt.columns[1:]].div(df_total, 0)*100

for n in df_rel3:
    for i, (cs, ab, pc, tot) in enumerate(zip(iotc_df.iloc[:, 2:].cumsum(1)[n],
                                              iotc_df[n], df_rel3[n], 
                                              df_total)):
        p1.text(tot + 30, i, f'{tot:,}', va='center', fontsize = 22)
        
        if i != 3:
            if pc < 40:
                continue
            else:
                p1.text(cs - ab/2 + 5, i, str(int(np.round(pc))) + '%', 
                        va='center', ha='center', rotation = 90, 
                        fontsize = 22)
        else:
            if pc < 30:
                continue
            else:
                p1.text(cs - ab/2, i, str(int(np.round(pc))) + '%', 
                        va='center', ha='center', fontsize = 22)
            

#-- 
df_total = wcpfc['Total Count'].astype(int)

wcpfc_order = [ 'Republic of Korea', 'Vanuatu', \
               'Fishing entity of Taiwan', 'China']
wcpfc.Country = wcpfc.Country.astype("category")
wcpfc.Country.cat.set_categories(wcpfc_order, inplace=True)
wcpfc = wcpfc.sort_values('Country')

wcpfc_plt = wcpfc.iloc[:,1:4]
p2 = wcpfc_plt.plot(x = 'Country', kind='barh', ax = f4_ax1, 
                    width=0.8,stacked = True, 
              mark_right = True, color = ['mediumseagreen', 'lightslategray'],
                    legend = False, xlabel = '', fontsize = 20)
p2.set_title(r"$\bf{" + 'd' + "}$" + ' Western and Central Pacific\nFisheries Commission', 
             fontsize = 30)
f4_ax1.set_yticklabels([])
f4_ax1.set_xlabel('Number of Vessels', fontsize = 23)
f4_ax1.tick_params(axis='both', which='major', labelsize=22)

p2.grid(False)

df_rel4 = wcpfc_plt[wcpfc_plt.columns[1:]].div(df_total, 0)*100
for n in df_rel4:
    for i, (cs, ab, pc, tot) in enumerate(zip(wcpfc.iloc[:, 2:].cumsum(1)[n], 
                                              wcpfc[n], df_rel4[n], 
                                              df_total)):
        p2.text(tot + 20, i, str(tot), va='center', fontsize = 22)
        
        if i == 0:
            if pc < 20:
                continue
        
            else:
                p2.text(cs - ab/2, i, str(int(np.round(pc))) + '%', 
                        va='center', ha='center', rotation = 90, 
                        fontsize = 22)
                
        elif i == 1:
            if pc < 20:
                continue
        
            else:
                p2.text(cs - ab/2 + 5, i, str(int(np.round(pc))) + '%', 
                        va='center', ha='center', 
                        rotation = 90, fontsize = 22)
                
        
        else:
            if pc < 20:
                continue
            else:
                p2.text(cs - ab/2, i, str(int(np.round(pc))) + '%', 
                        va='center', ha='center', fontsize = 22)
            
#--
df_total = iattc['total_count'].astype(int)

iattc.Country = iattc.Country.astype("category")
iattc.Country.cat.set_categories(wcpfc_order, inplace=True)
iattc = iattc.sort_values('Country')

iattc_plt = iattc.iloc[:,1:4].sort_values('Country', ascending = True)


p3 = iattc_plt.plot(x = 'Country', kind='barh',ax = f4_ax2, 
                    width=0.8, stacked = True,
              mark_right = True, legend = False, 
                    color = ['mediumseagreen', 'lightslategray'], 
                    xlabel = '', fontsize = 20)
p3.set_title(r"$\bf{" + 'e' + "}$" + ' Inter-American Tropical\nTuna Commission', 
             fontsize = 30)
p3.grid(False)
f4_ax2.set_yticklabels([])
f4_ax2.tick_params(axis='both', which='major', labelsize=22)

df_rel5 = iattc_plt[iattc_plt.columns[1:]].div(df_total, 0)*100
for n in df_rel5:
    for i, (cs, ab, pc, tot) in enumerate(zip(iattc.iloc[:, 2:].cumsum(1)[n], 
                                              iattc[n], df_rel5[n], 
                                              df_total)):
        p3.text(tot + 5, i, str(tot), va='center', fontsize = 22)
        
        if pc < 20:
            continue
        else:
            p3.text(cs - ab/2, i, str(int(np.round(pc))) + '%', 
                    va='center', ha='center', fontsize = 22)
            
            
#------------------------EEZ
mad_eez_count2['total'] = mad_eez_count2.sum(axis=1)
mad_eez_count2 = mad_eez_count2.sort_values('total', ascending=True)
mad_eez_count2.iloc[:,:-1].plot(kind='barh', stacked=True, 
                                legend=['eez_name'],ax =  f4_ax5,
                                color=[navy, purple, green, orange] , 
                                width = .7,  xlabel = '', fontsize = 20)
mad_labels = ['Republic\nof Korea', 'China', 'Seychelles', \
              'Fishing entity\nof Taiwan']
f4_ax5.set(yticklabels = mad_labels)
f4_ax5.set_ylabel('Indian Ocean', fontsize = 30)
f4_ax5.set_title(r"$\bf{" + 'a' + "}$" + ' Vessels with AIS\nin Study Area', 
                 fontsize = 30)
f4_ax5.legend(title='Location', loc='lower right', frameon=False, 
              fontsize = 22, bbox_to_anchor=(1.05, 0))
f4_ax5.tick_params(axis='both', which='major', labelsize=22)
plt.rcParams['legend.title_fontsize'] = 22
f4_ax5.grid(False)

#---
fp_eez_count2['total'] = fp_eez_count2.sum(axis=1)
fp_eez_count2 = fp_eez_count2.sort_values('total', ascending=True)
fp_eez_count2.iloc[:,:-1].plot(kind='barh', stacked=True, 
                               legend=False,ax = f4_ax3, color = [navy],
                               width = .7, xlabel = '', fontsize = 20)
f4_ax3.set_ylabel('Pacific Ocean', fontsize = 30)
f4_ax3.set_title(r"$\bf{" + 'c' + "}$" + ' Vessels with AIS\nin Study Area', 
                 fontsize = 30)
fp_labels = ['Republic\nof Korea', 'Vanuatu','Fishing entity\nof Taiwan',\
             'China']
f4_ax3.set(yticklabels = fp_labels)

f4_ax3.grid(False)
f4_ax3.tick_params(axis='both', which='major', labelsize=22)
fig4.subplots_adjust(hspace=.4, wspace=.5)

f4_ax4.xaxis.set_major_formatter(
        tkr.FuncFormatter(lambda y,  p: format(int(y), ',')))


# fig4.savefig('fig4_horiz.png', dpi=300, bbox_inches='tight')

plt.show()


# %%
