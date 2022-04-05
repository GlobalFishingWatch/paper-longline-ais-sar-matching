# %% [markdown]
# # Get relationship between vessel length from AIS and SAR
#
# import math
# from random import random
#
# import matplotlib.pyplot as plt
# import numpy as np
# %%
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import ais_sar_matching.sar_analysis as sarm

# %load_ext autoreload
# %autoreload 2

# %%
q = """

with fishing_vessels as (
select ssvid from `world-fishing-827.gfw_research.vi_ssvid_v20201101`
where on_fishing_list_best)


select
ssvid,
gfw_length,
sar_length ,
sar_length is not null as detected,
ssvid in (select * from fishing_vessels) as is_fishing,
region,
from proj_walmart_dark_targets.all_detections_and_ais_v20201207
where ssvid not in (select ssvid from `proj_walmart_dark_targets.all_mmsi_vessel_class` where final_vessel_class = 'gear')

-- eliminate vessels that might have been used by multiple names
and ssvid not in (select ssvid from `gfw_research.vi_ssvid_v20201101` where activity.overlap_hours_multinames > 24)

"""

df = sarm.gbq(q)

# %%
# filter to things that have lengths in both
df = df[(df.gfw_length > 0) & (df.sar_length > 0)]
df.head()

# %%
plt.figure(figsize=(10, 5))
sns.histplot(df.sar_length, label="lengths from SAR")
sns.histplot(df.gfw_length, label="lengths from GFW", color="orange")
plt.xlabel("length, m")
plt.legend()

# %% [markdown]
# ## The SAR lengths are much more widely distributed.
#

# %%
# scatter plot of length from AIS versus lenth from SAR
d = df[df.region == "indian"]
plt.scatter(d.gfw_length, d.sar_length, alpha=0.2, label="indian")
d = df[df.region == "pacific"]

plt.scatter(d.gfw_length, d.sar_length, alpha=0.2, label="pacific")
plt.xlabel("Length from AIS")
plt.ylabel("Length from SAR")
plt.xlim(0, 350)
# plt.ylim(0,350)
plt.plot([0, 350], [0, 350])
plt.title("length from SAR versus AIS")
plt.legend()
plt.show()

# %% [markdown]
# ### Take Aways:
#  - the SAR lengths are not normaly distributed around a 1:1 line
#  - there is a really wide spread in SAR length for a given AIS length
#
#
# ### Transorm data and do a linear regression
#  - do a box-cox tranformation, which transforms SAR lengths by (sar_length^lambda -1)/lambda -- note that we save the value of lambda below as the variable `fitted_lambda`
#  - fit a line y = Ax + B, where x is length in AIS and y is the length in SAR under box-cox transformation
#  - check to see if errors are normally distributed
#  - check for heteroskedasticity

# %%

# SAR
# do a box cox transformation of the length from sar:
# https://en.wikipedia.org/wiki/Power_transform
fitted_data, fitted_lambda = stats.boxcox(df.sar_length)

# AIS
# xfitted_data, xfitted_lambda = stats.boxcox(df.gfw_length)
# x = np.log(x)
x = df.gfw_length

# reshape vector
x = np.array(x).reshape(-1, 1)
# x = xfitted_data.reshape(-1, 1)
y = fitted_data.reshape(-1, 1)

# inspect new histograms
plt.figure(figsize=(10, 5))
# sns.histplot(y, label='transformed lengths from SAR')
sns.histplot(x, label="transformed lengths from GFW", color="orange")
plt.legend()


# Fitting the model
model = LinearRegression()
model.fit(x, y)

# Returning the R^2 for the model
model_r2 = model.score(x, y)
print("R^2: {0}".format(model_r2))

# %% [markdown]
# ### R2 is only .3
#
# SAR does predict length, but not that well.

# %%
# save the model variables in A and B, so A + BX = Y,
# where X = the length from AIS in meters
# and Y = the box cox transformation of SAR lenth

A = model.intercept_[0]
B = model.coef_[0][0]


# %%
df["fitted_data"] = fitted_data
df.head()

# %%
# Apply the box cox algorithm to make sure things are doing what we think
# fitted_data_test should equal fitted data
df["fitted_data_test"] = df.sar_length.apply(
    lambda x: (x ** fitted_lambda - 1) / fitted_lambda
)

# %%
df.head()

# %%
# Yep, the box cox transofrmation is doing what we expect

# %%
d = sarm.calculate_residuals(model, x, y)
d.head()


# %%
# plt.scatter(d.x.values, d.Residuals.values)

# %%
plt.plot(d.x.values, d.Predicted.values, color="grey")
plt.scatter(d.x.values, d.Actual.values, alpha=0.5)
plt.xlabel("AIS Length")
plt.ylabel("Box Cox transformd SAR length")
plt.title("Regression of AIS Length to \n Box-Cox Transformed SAR Length")


# %%
# Does it look like the errors are normally distributed?
sns.displot(d.Residuals)

# %% [markdown]
# #### It *looks* like the residuals are normally distributed. Let's do a fancy test to see if they are.

# %%
sarm.normal_errors_assumption(model, x, y)

# %% [markdown]
# ### Great -- a function I copied from the internet says my residuals are normally distributed. Now get the standard deviation of the residuals and save it.
#
#
# ### Now Check heteroscedasticity

# %%
d.Residuals.std()

# %%
# save this in a variable
stdev_residuals = d.Residuals.std()

# %%

# %%
# Okay... are these residuals distriburted normally?
# And what about heteroscedasticity?
# plot the squre of the difference... does it show a trend?
plt.scatter(d.x.values, (d.Actual.values - d.Predicted.values) ** 2, alpha=0.1)
plt.title("residuals squared")
plt.xlabel("GFW Length")


# %%
# To the eye, it *looks* like the errors are larger for smaller vessels,
# but because
# there are more smaller vessels than larger, it might just make the graph
# appear that way

# We will have to fit a line and see if there is a significant trend

# %%

# this is just another way to do the linear regression... shows the
# same results as the linear model from scikit-learn

# uncomment this to see all types of details about the model
# expr = 'Actual ~ x'
# olsr_results = smf.ols(expr, d).fit()
# print(olsr_results.summary())

# %%
# Let's see if the residuals have a trend.

d["x2"] = np.power(d.x, 2.0)
d["Residuals_2"] = (d.Actual.values - d.Predicted.values) ** 2

aux_expr = "Residuals_2 ~ x + x2"
Y, X = dmatrices(aux_expr, d, return_type="dataframe")
X = sm.add_constant(X)

aux_olsr_results = sm.OLS(Y, X).fit()
print(aux_olsr_results.summary())

# %% [markdown]
# ### No trend in the residuals
#
# The p value for the slope and squared terms are very insignificant...
# so the residuals don't appear to be increasing or decreasing, even though it looks
# like, on the graph I made, that they are decreasing

# %% [markdown]
# ### Applying this linear model back to the original data, and see if I get something that looks like the original data

# %%
d.head()


# %%
def produce_values(x):
    """given a length in AIS, produce a value for what it might be in SAR,
    with a random term"""
    predicted = A + B * x
    pred_with_random = predicted + np.random.normal(loc=0, scale=stdev_residuals)
    return pred_with_random


stdev_residuals = d.Residuals.std()

d["random_fit"] = d.x.apply(produce_values)

plt.scatter(d.x, d.random_fit, alpha=0.5, label="simulated result")
plt.scatter(d.x, d.Actual, alpha=0.5, label="actual result")
plt.plot(d.x.values, d.Predicted.values, color="grey", label="fit")
plt.title("AIS Length vs. SAR Length, actual and simulated")
plt.legend()
plt.xlabel("length, m, from AIS")
plt.ylabel("length, m, from SAR,\n box cox transformed")
plt.show()

# %% [markdown]
# #### The simulated lengths look pretty good.


# %% [markdown]
# ### Can we inverse the relationship between SAR Length and AIS length and use this equation to predict AIS length from SAR length?


# %%
d.head()


# %%

# what happens if we flip the function and try to predict the length in AIS
# from the SAR length? I have two functions here -- get_lengths_r,
# which includes a random term


def get_ais_lengths_from_sar_r(y):
    """predicted length in AIS, including a random term"""
    return (
        (y ** fitted_lambda - 1) / fitted_lambda
        - A
        + np.random.normal(loc=0, scale=stdev_residuals)
    ) / B


def get_ais_lengths_from_sar(y):
    """predicted length in AIS, not including a random term"""
    return ((y ** fitted_lambda - 1) / fitted_lambda - A) / B


#
df["back_prediteced_lengths"] = df.sar_length.apply(get_ais_lengths_from_sar)
df["back_prediteced_lengths_r"] = df.sar_length.apply(get_ais_lengths_from_sar_r)

# %%
# really just double checking that my formula is right
d["Preditected_again"] = d.x.apply(lambda x: A + B * x)

# %%
B

# %%
d.head()

# %%
df.head()

# %%
# Okay... this will be fun... let's back predict ais length from the sar lengths...


fl = fitted_lambda

plt.plot(
    [i for i in range(250)],
    [1 / B * ((i ** fl - 1) / fl - A) for i in range(250)],
    label="predicted sar length",
)
plt.plot(
    [i for i in range(250)],
    [1 / B * ((i ** fl - 1) / fl - A) + 2 / B * stdev_residuals for i in range(250)],
    label="predicted sar length +- 2 standard deviations",
    color="grey",
)
plt.plot(
    [i for i in range(250)],
    [1 / B * ((i ** fl - 1) / fl - A) - 2 / B * stdev_residuals for i in range(250)],
    color="grey",
)  #    label = 'predicted sar length - 2 standard deviations',


plt.ylim(-30, 200)
plt.scatter(
    df.sar_length.values, df.gfw_length.values, alpha=0.8, label="actual lengths"
)
plt.scatter(
    df.sar_length.values,
    df.back_prediteced_lengths_r,
    alpha=0.4,
    label="predicted lengths from SAR",
)
plt.xlabel("length from SAR")
plt.ylabel("length from AIS")
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Predicting AIS Length from SAR Length")


# %% [markdown]
# # Comments on predicting actual vessel length from the length in SAR:
#  - the actual equation gives lots of negative values, which isn't possible!
#  - the spread is really wide! But this equation above looks accurate.
