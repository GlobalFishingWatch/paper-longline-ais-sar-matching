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
# # This notebook includes the analysis to determine the number of scenes required to x percent of longline activity, and the code to generate figure 5

# %matplotlib inline
import os

import cartopy
import matplotlib as mpl
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
# %%
import numpy as np
import pandas as pd
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps as psm
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False

def gbq(q):
    return pd.read_gbq(q, project_id='world-fishing-827')


# %% [markdown]
# ### Each SAR Scene is ~470 km wide
#
# #### build a grid that is 4 degree by 4 degree, every week of longline fishing

# %%
q = '''
SELECT
floor(extract(dayofyear from date)/7) week,
floor(cell_ll_lat/4) lat_index,
floor(cell_ll_lon/4) lon_index,
sum(fishing_hours) fishing_hours
FROM `global-fishing-watch.gfw_public_data.fishing_effort_v2`
where date(date) between "2020-01-01" and "2020-12-31"
and geartype = 'drifting_longlines'
and fishing_hours >0
group by lat_index, lon_index,week
order by fishing_hours desc
'''

df = gbq(q)

# %%
len(df)

# %%
df.head()

# %%
df.fishing_hours.sum()/1e6

# %%
df['fishing_sum'] = df.fishing_hours.cumsum()/df.fishing_hours.sum()

# %%
df.head()

# %%
df.fishing_sum.max()


# %%
def get_images(rate,df):
    for index, row in df.iterrows():
        if row.fishing_sum>rate:
            break
    print(f"images to capture {rate} of the fishing: {index}")
    return index

for rate in [.05,.1,.2,.25,.5,.75,1]:
    get_images(rate,df)

# %%
plt.plot(df.index, df.fishing_sum)
plt.title("Number of Images to Capture\n Fraction of Fishing Activity")
plt.xlabel("Number of Images")
plt.ylabel("Fraction")


# %%
def map_fraction(df, rate,max_value=30):
    index = get_images(rate,df)

    d = df[df.index<index].copy()
    len(d)
    d['one'] = 1
    cells_to_monitor = pyseas.maps.rasters.df2raster(d, 'lon_index', 'lat_index',
                               'one', xyscale=.25,
                                per_km2=False, origin = 'lower')


    raster=np.copy(cells_to_monitor)
    plt.rc('text', usetex=False)
    pyseas._reload()
    fig = plt.figure(figsize=(14, 7))
    norm = mpcolors.Normalize(vmin=0, vmax=max_value)
    raster[raster==0]=np.nan
    with plt.rc_context(psm.styles.dark):
        ax, im, cb = psm.plot_raster_w_colorbar(raster ,
                                           r"images per cell ",
                                           cmap='presence',
                                          norm=norm,
                                          cbformat='%.0f',
                                          projection = cartopy.crs.EqualEarth(central_longitude=-157),
                                          origin='lower',
                                          loc='bottom')

        psm.add_countries()
        psm.add_eezs()
        ax.set_title(f'To image {int(rate*100)}% of Longline Activity in 2020,
                     {index} Scenes', pad=10, fontsize=15 )
        psm.add_figure_background()
        psm.add_logo(loc='lower right')

        rate_str = str(rate).replace("0.","").replace
        #plt.savefig(f"images/cells_to_moniotor_longlines_{rate}.png",
                     dpi=300, bbox_inches = 'tight')

# %%
index = get_images(0.5,df)

d = df.loc[df.index<index].copy()
print(len(d))
d['one'] = 1

cells_to_monitor = pyseas.maps.rasters.df2raster(d, 'lon_index', 'lat_index',
                           'one', xyscale=.25,
                            per_km2=False, origin = 'lower')


raster=np.copy(cells_to_monitor)
plt.rc('text', usetex=False)


# %%
map_fraction(df, .5)

# %%
map_fraction(df, .1)

# %%
fishing = pyseas.maps.rasters.df2raster(df, 'lon_index', 'lat_index',
                           'fishing_hours', xyscale=.25,
                            per_km2=True, origin = 'lower')

# %%
fishing.max()

# %%
raster=np.copy(fishing)*1000
plt.rc('text', usetex=False)
pyseas._reload()
fig = plt.figure(figsize=(14, 7))
norm = mpcolors.Normalize(vmin=0, vmax=300)
raster[raster==0]=np.nan
with plt.rc_context(psm.styles.dark):
    ax, im, cb = psm.plot_raster_w_colorbar(raster ,
                                       r"fishing hours per 1000 km2 ",
                                       cmap='fishing',
                                      norm=norm,
                                      cbformat='%.0f',
                                      projection = 'global.pacific_centered',
                                      origin='lower',
                                      loc='bottom')

    psm.add_countries()
    psm.add_eezs()
    ax.set_title('Pelagic Longline Fishing in 2020', pad=10, fontsize=16 )
    psm.add_figure_background()
#     gl = pyseas.maps.add_gridlines()
    #pyseas.maps.core._last_projection = 'global.pacific_centered'
    fig.patch.set_facecolor('white')
    psm.add_logo(loc='lower right')


    plt.savefig("longline_fishing_2020.png", dpi=300, bbox_inches = 'tight')

# %%

# %%
index

# %%
index = get_images(.5,df)

d = df[df.index<index].copy()
d['one']= 1

# %%
d.head()

# %%
week_images = []
for week in range(52):
    cells_to_monitor = pyseas.maps.rasters.df2raster(d[d.week==week],
                                                     'lon_index', 'lat_index',
                                                     'one', xyscale=.25,
                                                     per_km2=False,
                                                     origin = 'lower')
    week_images.append(cells_to_monitor)

# %%
week_images[51]

# %% [markdown]
# ### Figure 5 for paper

# %%
plt.rcParams['axes.facecolor'] = 'white'


r = 11 #rows
c = 5 #columns

heights = [a.shape[0] for a in week_images[:r]]
widths = [a.shape[1] for a in week_images[:c]]

fig_width = 15  # inches
fig_height = fig_width * sum(heights) / sum(widths)


fig5 = plt.figure(figsize=(fig_width, fig_height))

spec5 = fig5.add_gridspec(ncols=c, nrows=r, height_ratios=heights)
i = 0
with psm.context(pyseas.styles.light):
    norm = mpcolors.LogNorm(vmin=.1, vmax=1)
    for row in range(r):
        for col in range(c):
            raster=np.copy(week_images[i])
            ax, im = psm.plot_raster(raster,
                                    spec5[row,col],
                                    cmap='fishing',
                                    norm=norm,
                                    projection = cartopy.crs.EqualEarth(central_longitude=-157),
                                    origin='lower')
            pyseas.maps.core._last_projection = cartopy.crs.EqualEarth(central_longitude=-157)
            psm.add_land(facecolor="lightgrey", edgecolor = 'none')
            i+=1
            if i == 52:
                break
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1,
                            bottom=0, top=1)
fig5.savefig("Fig5.png",dpi=300,bbox_inches='tight')

# %%
fig5.savefig("Fig5_3_50.png",dpi=300,bbox_inches='tight')

# %%
plt.rcParams['axes.facecolor'] = 'white'

fig6 = plt.figure(figsize=(20, 18))

spec6 = fig6.add_gridspec(ncols=10, nrows=5)

rate = .50
max_value = 30

index = get_images(rate,df)

d = df[df.index<index].copy()
# len(d)
d['one'] = 1
cells_to_monitor = pyseas.maps.rasters.df2raster(d, 'lon_index', 'lat_index',
                           'one', xyscale=.25,
                            per_km2=False, origin = 'lower')


raster=np.copy(cells_to_monitor)
plt.rc('text', usetex=False)
pyseas._reload()
norm = mpcolors.Normalize(vmin=0, vmax=max_value)
raster[raster==0]=np.nan
with plt.rc_context(psm.styles.light):
    ax, im = psm.plot_raster(raster,
                             spec6[0:3, :],
                             cmap='presence',
                             projection = cartopy.crs.EqualEarth(central_longitude=-157),
                             norm = norm,
                             origin='lower')
    psm.add_countries()
    psm.add_land(facecolor="lightgrey", edgecolor = 'none')
    ax.set_title(f' {index:,} Scenes to Image {int(rate*100)}% of Longline Activity in 2020',
                 pad=10, fontsize=28)
    psm.add_figure_background()
    pyseas.maps.core._last_projection = cartopy.crs.EqualEarth(central_longitude=-157)

ax3 = fig6.add_subplot(spec6[3, 5:-1])
ax3.axis('off')
cbar = fig6.colorbar(im, ax= ax3,
                    orientation = 'horizontal',
#                     location = 'bottom',
                        label = 'Number of images'
#                   fraction=.05,
#                   shrink = .5,
#                   aspect=40,
                 )
cbar.ax.tick_params(labelsize=20)
cbar.ax.set_xlabel('Number of images', size=24, labelpad=5)
# rate_str = str(rate).replace("0.","").replace

        #################

ax2 = fig6.add_subplot(spec6[3:, 1:5])

ax2.plot(df.index, df.fishing_sum, linewidth = 4)
ax2.set_xlabel("Number of images needed", size = 24, labelpad=10)
ax2.set_ylabel("Fraction of\n longline activity", size = 24, labelpad=15)
ax2.xaxis.set_major_formatter(
        tkr.FuncFormatter(lambda y,  p: format(int(y), ',')))
# start, end = ax2.get_xlim()
# ax2.xaxis.set_ticks(np.arange(0, end, 5000))
ax2.plot([3820,3820],[0,.5], '--', c = 'dimgray')
ax2.text(4000,0.2,'3,820 scenes\n50% of fishing',size = 20)
ax2.tick_params(labelsize=20)
ax2.grid(False)

plt.text(0.25, .86, 'a', fontsize=22, fontweight = 'bold',
         transform=plt.gcf().transFigure)
plt.text(0.21, .42, 'b', fontsize=22, fontweight = 'bold',
         transform=plt.gcf().transFigure)

fig6.set_facecolor("white")

# %%

# %%
fig6.savefig("Fig6.png",dpi=300,bbox_inches='tight')
