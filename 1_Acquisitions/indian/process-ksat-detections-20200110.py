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

# # process KSAT detections
#
# process `.kmz` files from KSAT for matching scripts
#
# updated 2020-01-10

# ! ls

import os
import pandas as pd
import glob
import zipfile
import kml2geojson as k2g
import json
import datetime
from shapely import geometry
import fiona 
import geopandas as gpd

pd.set_option('display.max_colwidth', -1)

# ## process detections

path = './csv' # use your path
filenames = glob.glob(path + "/*.csv")

# +
detections = []

for f in glob.glob("./csv/*.csv"):
    df = pd.read_csv(f, index_col=None, header=0)
    detections.append(df)

df = pd.concat(detections, axis=0, ignore_index=True)
df
# -

# ## process footprints

# +
d = []

for f in glob.glob('./kmz/*.kmz'):
    
    filename = f.split('.')[1].split('/')[2].split('.')[0]

    c = 'cp ./kmz/{}.kmz ./kml/tmp.zip'.format(filename)
    os.system(c)
    
    with zipfile.ZipFile('./kml/tmp.zip', 'r') as zip_ref:
        zip_ref.extractall('./kml/')

    # ! rm ./kml/tmp.zip
    # ! rm ./kml/vd.png
    
    c = 'k2g ./kml/{}.kml ./geojson'.format(filename)
    os.system(c)
    
    with open('./geojson/{}.geojson'.format(filename) ,'rU') as f:
        j =  f.read()
        j = json.loads(j)
        
    footprint = []

    for i in range(0, len(j['features']), 1):
        if j['features'][i]['geometry']['type'] == 'Polygon':
            coords = j['features'][i]['geometry']['coordinates']
            footprint.append(coords)

    footprint[0][0]
    
    coords = [x[:-1] for x in footprint[0][0]]
    
    poly = geometry.Polygon([[p[0], p[1]] for p in coords])

    d.append({'scene_id':filename, 'footprint':poly})
    
#     c = 'rm -r ./geojson/*'
#     os.system(c)   
    
df_footprint = pd.DataFrame(d)
df_footprint['scene_id'] = df_footprint.scene_id.apply(lambda x: x[0:len('RS2_20190903_041744_0075_DVWF_HH_SCS_754024_9569_29818064')])
df_footprint
# -

# ## merge/clean tables & BQ load

df_to_bq = pd.merge(df, df_footprint, left_on='ProductId', right_on='scene_id')
df_to_bq.head()

df_to_bq = df_to_bq.drop(columns = ['description', 'Property', 'ProductId'])
df_to_bq = df_to_bq.rename(columns = {'X':'lon','Y':'lat'})
df_to_bq.head()

df_to_bq.dtypes

df_to_bq['ProductStartTime'] = pd.to_datetime(df_to_bq['ProductStartTime'])

df_to_bq['ProductStopTime'] = pd.to_datetime(df_to_bq['ProductStopTime'])

df_to_bq.dtypes

df_to_bq.to_gbq('paper_longline_ais_sar_matching.ksat_detections_ind_v20200110', 
                project_id='global-fishing-watch', if_exists='replace')


df_to_bq.to_csv('ksat-detections-ind-v20200110.csv')


