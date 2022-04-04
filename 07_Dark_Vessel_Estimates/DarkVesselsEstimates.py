# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Estimate the Number of Dark Vessels
#
# 1. Model the relationship between SAR length and AIS length
# 2. Model the recall curve as a proxy for detection rate
# 3. Estimate dark vessels as a function of length, recall, and observations

# +
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyseas.maps as psm
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy import newaxis

plt.rcParams["axes.grid"] = False
import pickle
import warnings

import proplot as pplt
import scipy
from scipy.special import erfinv, gammaln
# %matplotlib inline
from scipy.stats import binom, gaussian_kde, lognorm, poisson
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rc("legend", fontsize="14")
plt.rc("legend", frameon=False)

purple = "#d73b68"
navy = "#204280"
lighter_navy = "#5a7fc4"
orange = "#f68d4b"
gold = "#f8ba47"
green = "#ebe55d"

fishing_color = gold  #'orange'
nonfishing_color = lighter_navy  #'blue'
dark_vessel_color = "#545454"

# %load_ext autoreload
# %autoreload 2


def gbq(q):
    return pd.read_gbq(q, project_id="world-fishing-827")


# Important! This is the threshold that we accpet matches
# We found it was between 5e-6 and 1e-4. This is the mean
# of these two values, and in theory we should also test
# how much the model changes based on changing this value
matching_threshold = 2.5e-5


# -

# # Get all detections and non-gear ssvid
#
# Fields:
#  - ssvid: vessel mmsi. If it is null, it is a detection that could not match to anything.
#  - score: matching score for detection to ssvid. Thereshold for accepting a match is between 5*10^-6 and 1*10^-4.
#  - detect_id: unique identifier for SAR detection. null if there is none
#  - gfw_length: If vessel has ssvid, the length from the GFW vessel database.
#  - sar_length: If there is a detection, the lenght reported by SAR
#  - is_fishing: If the vessel is a fishing vessel
#  - likelihod_in: chance (from 0-1) that a vessel is within the scene.
#  - region: either "indian" or "pacific"
#  - within_fooptrint_5km: Is the likely position more than 5 km within the scene?
#  - match_review: is "yes", "ambiguous", "maybe", and "no". Select "yes" for definite matches, which is used for comparing sar length with gfw_length
#
#
# Notes:
#  - a single row can have a ssvid and a detect_id, but represent two different vessels. If the score < 5e-6 it almost definitely represents two distinct vessels (one in AIS that was not detected by SAR and one vessel detected by SAR that was not broadcasting).
#  - there are lots and lots of vessels with a likelihood in <.5
#  - if the sar detection matches to no AIS vessels, it has a score of 0
#  - the same ssvid can appear in multiple scenes
#
# To select all AIS vessels likely in a scene:
#  - df[df.likelihod_in>.5]
#
# To select all AIS to detection pairs that are almost definitely matches:
#  - df[df.match_review == "yes"]
#
#
# Select all SAR detections that represent non-broadcasting vessels, where `matching_threshold` is a value between 5e-6 and 1e-4:
#  - df[(df.score < matching_threshold)&(~df.detect_id.isna())]
#
# Select AIS vessels that likely matched to SAR detections:
# - df[df.score > matching_threshold]

# +
## Helper Functions for quantile regression and getting results


def fit_line(x, y, q=0.5):
    """Fit line to specified quantile."""
    model = smf.quantreg("y ~ x", {"x": x, "y": y})
    fit = model.fit(q=q)
    return [
        q,
        fit.params["Intercept"],
        fit.params["x"],
    ] + fit.conf_int().loc["x"].tolist()


def get_y(a, b, x):
    return a + b * x


def get_x(a, b, y):
    # The inverse function x(y) to be used in the prob
    return (y - a) / b


# NOTE: Because the above is an inverse function, it will
# return negative values for very small lenghts (<20m),
# so below these lenghts the model is invalid.

# +

q = """
SELECT
  ssvid,
  score,
  detect_id,
  scene_id,
  gfw_length,
  sar_length,
  is_fishing,
  likelihood_in,
  region,
  within_footprint_5km,
  match_review
FROM
  `world-fishing-827.proj_walmart_dark_targets.all_detections_and_ais_v20210427`
WHERE
  vessel_type not in ('gear','duplicate')
  OR detect_id IS not NULL
"""

df = pd.read_gbq(q, project_id="world-fishing-827")
# -

# save here to access offline
df.to_csv("all_detections_and_vessels.csv")

df = pd.read_csv("all_detections_and_vessels.csv")

for region in ["indian", "pacific"]:
    print(
        f"sar detections in {region}: {len(df[(df.region==region)&(~df.sar_length.isna())])}"
    )

# # `df_m`
# A dataframe for definite AIS to SAR matches, and used to compute SAR to GFW lengths

df_m = df[df.match_review == "yes"]  # only high confidence, unambiguous matches

# # `df_r`
# Vessels in AIS that definitely appeared in the scene. This is for vessels with a more than 99% chance of appearing in a scene, and it is used to calcluate recall as a function of vessel length.

df_r = df[
    (~df.ssvid.isna()) & (df.likelihood_in > 0.99)
]  # Is a vessel and alsmost definitely in the scene


# ## Calculate detection rate per vessel for large (>60m) and small (<60m) vessels
#
# The 60m cutoff is a bit arbitrary, but there are very few vessels between 60 and 100, so you could choose a higher number and get just about the same results. We also find a roughly linear relationship between detection rate and vessel size
#
# `df_grouped` groups df_r by ssvid, and then calculates `detected_t`, which is the detection rate for that ssvid. An ssvid that appeared in four scenes and was detected once by SAR (as measured by a score > matching_threshold) has detected_t = .25
#
# `df_grouped` is then divided into `df_small` -- vessels under 60 meters -- and `df_large`, vessels > 60 meters. The detection rate for df_large is averaged to get an average detection rate for large vessels.
#
# `detect_t` for vessels under 60 meters is later used to fit a line for vessels under this size.

# +
def get_df_grouped(df_r, matching_threshold):
    df_r.loc[:, "detected_t"] = df_r.score.apply(
        lambda x: 1 if x > matching_threshold else 0
    )
    df_r.loc[:, "detected"] = df_r.score.apply(
        lambda x: True if x > matching_threshold else False
    )
    # So, this treats each vessel seperately.
    # If the vessel appears in several scenes
    df_grouped = df_r.groupby("ssvid").mean()
    return df_grouped


df_grouped = get_df_grouped(df_r, matching_threshold)

max_size = 60
# Length greater/smaller than 60 meteres
df_small = df_grouped[df_grouped.gfw_length < max_size]
df_large = df_grouped[df_grouped.gfw_length > max_size]

max_detect_rate = df_large.detected_t.sum() / len(df_large)

print(f"maximum detection rate: {max_detect_rate:.03f}")
print(f"vessels detected >{max_size}m:  {df_large.detected_t.sum():g}")
print(f"total vessels >{max_size}m:     {len(df_large)}")
print(f"total vessels <{max_size}m:     {len(df_small)}")

# -


# ### Calculate binned detection rates.
#
# Calculate the average detection rate for all vessels between 10-20 meters, then between 15-25 meters, and so on (10m wide, step by 5m). This is not used for analysis, but rather to compare with the quantile regression.

# +


def bin_data(df, width=16, bins=50, median=False):
    """calculates the average detected_t, binned at width meters, moving
    by half window"""
    x_bin5, y_bin5 = [], []
    y_bin5std, x_bin5std = [], []
    half = width / 2
    for i in range(bins):
        cond = (df.gfw_length >= i * half) & (df.gfw_length < i * half + width)
        d = df[cond]
        if len(d) == 0:
            continue
        if median:
            y_bin5.append(d.detected_t.median())
        else:
            y_bin5.append(d.detected_t.mean())
        y_bin5std.append(d.detected_t.std() / np.sqrt(len(d)))
        x_bin5.append(d.gfw_length.mean())
        x_bin5std.append(d.gfw_length.std())
    return np.array(x_bin5), np.array(y_bin5), np.array(x_bin5std), np.array(y_bin5std)


# get averaged values every 10 vessels
# this is used to plot the average detection rate.
x_bin5, y_bin5, x_bin5std, y_bin5std = bin_data(df_r, width=10, median=False)
# -


# # Minimization Approach

# ## Compute the Detection Probility Vector

# +
def compute_bins(lmin, lmax, n):
    return np.linspace(lmin, lmax, n + 1, endpoint=True)


def compute_lengths(bins):
    return 0.5 * (bins[1:] + bins[:-1])


def compute_d(lengths, min_p=0.01, max_p=1.0, df_small=df_small):
    x = df_small.gfw_length.values
    y = df_small.detected_t.values
    q, a, b, lb, ub = fit_line(x, y, 0.5)
    return np.clip(get_y(a, b, lengths), min_p, max_p)


# -

# ## Compute the Matrix relating GFW to SAR Lengths

# +
class LComputer:
    """Computes a matrix relating AIS reported lengths to measured lengths

    We compute a matrix `L` that approximates the probability of measuring some
    length using SAR given the actual, AIS reported length. That is:

        s = L @ a

    Where `a` is a vector of the actual number of vessels at each lengths and
    `s` is the expected number of vessels at each length *as measured by SAR*.

    Primary entry points are `__call__`, which computes the `L` matrix and
    `plot_fits` which plots some example fits to show how well the approximation
    is performing.

    Parameters
    ----------
    sar_length, gfw_length : 1D np.array of float
        Matched pairs of GFW (true) and SAR (measured) lengths
    """

    lookup_table_size = 300
    shape_range = (0.05, 1)
    scale_range = (10, 500)
    quantiles = [0.333, 0.667]

    def __init__(self, gfw_length, sar_length):
        self.gfw_length = gfw_length
        self.sar_length = sar_length
        self._fit_quantiles()
        self.lookup_table = self._create_lookup_table()

    def _fit_quantiles(self):
        # Fit lines to `self.quantiles` using quantile
        # regression. The resulting fits are stored in self._models.
        mdls = [fit_line(self.gfw_length, self.sar_length, q) for q in self.quantiles]
        self._models = pd.DataFrame(mdls, columns=["q", "a", "b", "lb", "ub"])

    def _create_lookup_table(self):
        # We need to be able to efficiently find shape and scale values
        # for a lognormal distribution based on a set of quantile values.
        # We create a lookup table relating quantiles to shape and scale,
        # in order to do this quickly later.
        m = self.lookup_table_size
        M = np.empty([m, m, len(self.quantiles)])
        M.fill(np.nan)
        self._shape_vals = np.linspace(*self.shape_range, num=m, endpoint=True)
        self._scale_vals = np.linspace(*self.scale_range, num=m, endpoint=True)
        for i, shape in enumerate(self._shape_vals):
            for j, scale in enumerate(self._scale_vals):
                M[i, j] = self._compute_quantiles(shape, scale)
        return M

    def _compute_quantiles(self, shape, scale):
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html
        sigma = shape
        mu = np.log(scale)
        return [
            np.exp(mu + np.sqrt(2 * sigma ** 2) * erfinv(2 * p - 1))
            for p in self.quantiles
        ]

    def __call__(self, bin_edges, normalize=False):
        """Compute the `L` matrix relating measured SAR to actual lengths

        Parameters
        ----------
        bin_edges : sequence of float
            Designates the edges of the length bins to compute `L` for.
            If there are `n + 1` values in bin_edges, then there will
            be `n` bins.
        normalize : bool, optional
            If True, ensure that `L` is normalized over SAR lengths at each
            AIS length.

        """
        n = len(bin_edges) - 1
        L = np.zeros([n, n])
        for i, l in enumerate(compute_lengths(bin_edges)):
            cdf = self._dist_at(l).cdf(bin_edges)
            L[:, i] = cdf[1:] - cdf[:-1]
        if normalize:
            L /= L.sum(axis=0, keepdims=True)
        return L

    def _dist_at(self, l):
        # Return the CDF at length `l`
        qvals = [
            get_y(self._models.a[i], self._models.b[i], l)
            for (i, _) in enumerate(self.quantiles)
        ]
        shape, scale = self._lookup_quantiles(qvals)
        return lognorm(shape, loc=0, scale=scale)

    def _lookup_quantiles(self, qvals):
        # Lookup shape and scale values based on quantile values.
        m = self.lookup_table_size
        # `delta` is an m X m X 2 table, where the first two dimension
        # are indexed by shape and scale (see below) and the last gives
        # the difference with the tercies we are trying to find.
        delta = self.lookup_table - [[qvals]]
        # We find the minimum value of `eps`, the square error.
        # `eps` = delta_q333 ** 2 + delta_q667 ** 2
        eps = (delta ** 2).sum(axis=2)
        # This is a numpy trick to find the row and column of the minimum
        # value in an array.
        flat_ndx = np.argmin(eps)
        i, j = flat_ndx // m, flat_ndx % m
        # The row and column in the lookup table (and thus `delta`)
        # correspond to our stored shape and scale value, so look
        # those up and return them.
        return self._shape_vals[i], self._scale_vals[j]

    def plot_fits(self):
        """Plot some example plots for evaluating the fit of `L`"""
        for low, high, cntr, mask in self._get_ranges():
            y_dist = (self.sar_length - self.gfw_length)[mask] + cntr
            dist = self._dist_at(cntr)

            _ = plt.hist(y_dist, bins=20, density=True)
            plt.xlim(0)
            l = np.arange(1, 300)
            plt.plot(l, dist.pdf(l))
            plt.title(f"GFW Length: {low}m-{high}m")
            plt.ylabel("density")
            plt.xlabel("SAR length")
            plt.show()

    def _get_ranges(self):
        # Return appropriate ranges for plotting some fits over.
        for low in range(20, 300, 20):
            high = low + 20
            mask = (self.gfw_length > low) & (self.gfw_length < high)
            if mask.sum() < 20:
                continue
            cntr = (low + high) / 2
            yield low, high, cntr, mask


class LComputerQuartiles(LComputer):
    """Compute 'L' based on a best fit of quartiles"""

    quantiles = [0.25, 0.50, 0.75]


class LComputerIQD(LComputerQuartiles):
    """Compute 'L' based on a best fit of median and IQD"""

    def _create_lookup_table(self):
        M = super()._create_lookup_table()
        # Convert quartiles to median and inter-quartile distance
        median = M[:, :, 1]
        IQD = M[:, :, 2] - M[:, :, 0]
        return np.concatenate([median[:, :, np.newaxis], IQD[:, :, np.newaxis]], axis=2)

    def _lookup_quantiles(self, qvals):
        # Convert quartiles to median and inter-quartile distance
        median = qvals[1]
        IQD = qvals[2] - qvals[0]
        return super()._lookup_quantiles([median, IQD])


compute_L = LComputer(df_m.gfw_length.values, df_m.sar_length.values)
compute_L.plot_fits()
# compute_L = LComputerQuartiles(df_m.gfw_length.values, df_m.sar_length.values)
# compute_L.plot_fits()
# compute_L = LComputerIQD(df_m.gfw_length.values, df_m.sar_length.values)
# compute_L.plot_fits()
# -

# ## Wrapper Around Optimizer

# +
def compute_o(df, lengths, region="none"):
    valid = df[~df.sar_length.isnull()]
    if region == "none":
        sar_lengths = valid.sar_length
    else:
        sar_lengths = valid.sar_length[valid.region == region]
    indices = np.clip(np.searchsorted(lengths, sar_lengths), 0, n_bins - 1)
    o = np.zeros(n_bins)
    for i in indices:
        o[i] += 1
    return o


def infer_vessels(o, d, L, cf, maxfun=1e6):
    def objective(x):
        """Combined Kolmogorov–Smirnov and squared error of total counts

        Note that unlike some earlier objectives, this uses the classic
        KS error operating on distributions. This results on the two
        components of the objective being indepent, with KS error
        controlling the shape and the squared error controlling the
        amplitude.
        """
        e = L @ (d * cf(x))
        T_e = e.sum()
        T_o = o.sum()
        # Compute
        D_e = e / T_e
        D_o = o / T_o
        KS = abs(np.cumsum(D_e) - np.cumsum(D_o)).max()
        SE = (T_e - T_o) ** 2
        return KS + SE

    guess = cf.guess(o)
    constraint = scipy.optimize.LinearConstraint(
        np.identity(len(guess)), cf.lower_bounds, cf.upper_bounds
    )

    return scipy.optimize.minimize(
        objective,
        guess,
        constraints=[constraint],
        options=dict(maxfun=maxfun, maxiter=1000),
    )


# -
# ##### Infer Lengths

# +

lmin, lmax = 0, 400
n_bins = 400
bins = compute_bins(lmin, lmax, n_bins)
lengths = compute_lengths(bins)
compute_L = LComputer(df_m.gfw_length.values, df_m.sar_length.values)
L = compute_L(bins)

## Caculate d, the detection rate
df_grouped = get_df_grouped(df_r, matching_threshold)
df_small = df_grouped[df_grouped.gfw_length < max_size]
df_large = df_grouped[df_grouped.gfw_length > max_size]
max_detect_rate = df_large.detected_t.sum() / len(df_large)

d = compute_d(lengths, max_p=max_detect_rate, df_small=df_small)
# max_detect_rate was calculated as detection rate of vessels > 60m


# Coupling Matrix
model_bin_size = 5
assert n_bins % model_bin_size == 0
C = np.zeros([n_bins, n_bins // model_bin_size])
for i in range(n_bins):
    C[i, i // model_bin_size] = 1.0 / model_bin_size


class BinCouplingFunction:
    def __init__(self, n_bins, rel_bin_size, l_min, l_max):
        assert n_bins % rel_bin_size == 0
        self.out_bins = compute_bins(l_min, l_max, n_bins)
        self.lower_bounds = np.zeros(n_bins // rel_bin_size)
        self.upper_bounds = np.empty(n_bins // rel_bin_size)
        self.upper_bounds.fill(np.inf)
        #         self.upper_bounds[0:4] = np.zeros(4)
        self.n_bins = n_bins
        self.rel_bin_size = rel_bin_size
        self.C = self._compute_C(n_bins, rel_bin_size)

    def _compute_C(self, n_bins, rel_bin_size):
        C = np.zeros([n_bins, n_bins // rel_bin_size])
        for i in range(n_bins):
            C[i, i // rel_bin_size] = 1.0 / rel_bin_size
        return C

    def __call__(self, x):
        return self.C @ x

    def guess(self, x):
        return self.C.T @ x


# TODO: note there some assumptions here that base bin size is always 1 m
class BinCouplingFunctionLarge:
    def __init__(self, n_bins, rel_bin_size, l_min, l_max, min_vessel_sz=20):
        assert n_bins % rel_bin_size == 0
        self.out_bins = compute_bins(l_min, l_max, n_bins)
        self.lower_bounds = np.zeros(n_bins // rel_bin_size)
        self.upper_bounds = np.empty(n_bins // rel_bin_size)
        self.upper_bounds.fill(np.inf)
        # Sets upper bound to be zero for sizes below min_vessel_sz
        self.upper_bounds[: min_vessel_sz // rel_bin_size] = 0
        self.n_bins = n_bins
        self.rel_bin_size = rel_bin_size
        self.C = self._compute_C(n_bins, rel_bin_size)

    def _compute_C(self, n_bins, rel_bin_size):
        C = np.zeros([n_bins, n_bins // rel_bin_size])
        for i in range(n_bins):
            C[i, i // rel_bin_size] = 1.0 / rel_bin_size
        return C

    def __call__(self, x):
        return self.C @ x

    def guess(self, x):
        return self.C.T @ x


class AISPriorCouplingFunction(BinCouplingFunction):
    default_max_augment_len = 40.0
    lower_bounds = np.array([0, 0, 0])
    upper_bounds = np.array([np.inf, np.inf, np.inf])

    def __init__(self, n_bins, rel_bin_size, l_min, l_max, df):
        dvs = df
        assert n_bins % rel_bin_size == 0
        self.rel_bin_size = rel_bin_size
        self.in_bins = compute_bins(l_min, l_max, n_bins // rel_bin_size)
        self.out_bins = compute_bins(l_min, l_max, n_bins)

        base, _ = np.histogram(dvs.gfw_length, bins=self.in_bins)
        self.base = base / base.sum()
        self.C = self._compute_C(n_bins, rel_bin_size)

    def __call__(self, x):
        n, alpha, max_aug_len = x
        scale = np.maximum(1 - compute_lengths(self.in_bins) / max_aug_len, 0)
        binned = n * self.base * (1 + alpha * scale)
        return self.C @ binned

    def guess(self, x):
        return [x.sum(), 0, self.default_max_augment_len]


# +
# Select bin coupling function here for everywhere below

bin_coupling_func = BinCouplingFunctionLarge(n_bins, 5, lmin, lmax)
# bin_coupling_func = BinCouplingFunction(n_bins, 5, lmin, lmax)

# +
# vessels not in the sample and likely in the scenes

results_byregion = {}
for region in ["indian", "pacific"]:
    # detections in the region under the scoring threshold
    # non null sar length means it is a sar detection
    o = compute_o(
        df[
            (df.region == region)
            & (df.score < matching_threshold)
            & (~df.sar_length.isna())
        ],
        lengths,
    )
    results_byregion[region] = infer_vessels(o, d, L, bin_coupling_func)
    print(f"{region} has {results_byregion[region].x.sum():.0f} dark vessels")
# -


# # Updated Figure 3 and 4


# +
# following is to make a density plot
# drawing on seaborn

from seaborn import kdeplot

dark_distribution = {}
ais_distribution = {}

for region in ["pacific", "indian"]:

    all_sar = results_byregion[region].x

    best_lengths = []
    for j, v in enumerate(all_sar):
        v = int(round(v))
        for z in range(v):
            best_lengths.append(j * model_bin_size + model_bin_size / 2)

    my_data = best_lengths
    my_kde = kdeplot(my_data)
    line = my_kde.lines[0]
    x_all, y_all = line.get_data()
    plt.clf()

    ddark = df[(df.region == region) & (df.score < matching_threshold)]
    my_data = best_lengths
    my_kde = kdeplot(ddark.sar_length)
    line = my_kde.lines[0]
    x_sar, y_sar = line.get_data()
    plt.clf()

    dark_distribution[region] = (
        x_all,
        y_all * len(best_lengths),
        x_sar,
        y_sar * len(ddark),
    )

    d2 = df[~(df.ssvid.isna()) & (df.region == region) & (df.likelihood_in > 0.5)]
    my_kde = kdeplot(list(d2.gfw_length))
    line = my_kde.lines[0]
    x_all, y_all = line.get_data()
    plt.clf()

    dd = d2[d2.score > matching_threshold]
    my_kde = kdeplot(list(dd.sar_length))
    line = my_kde.lines[0]
    x_sar, y_sar = line.get_data()
    plt.clf()

    ais_distribution[region] = x_all, y_all * len(d2), x_sar, y_sar * len(dd)


# +
# all_sar = results_combined.x
all_sar = results_byregion["indian"].x + results_byregion["pacific"].x


best_lengths = []
for j, v in enumerate(all_sar):
    v = int(round(v))
    for z in range(v):
        best_lengths.append(j * model_bin_size + model_bin_size / 2)

my_data = best_lengths
my_kde = kdeplot(my_data)
line = my_kde.lines[0]
x_all, y_all = line.get_data()
plt.clf()

ddark = df[(df.score < matching_threshold)]
my_data = best_lengths
my_kde = kdeplot(ddark.sar_length)
line = my_kde.lines[0]
x_sar, y_sar = line.get_data()
plt.clf()

dark_distribution["all"] = x_all, y_all * len(best_lengths), x_sar, y_sar * len(ddark)


d2 = df[~(df.ssvid.isna()) & (df.likelihood_in > 0.5)]
my_kde = kdeplot(list(d2.gfw_length))
line = my_kde.lines[0]
x_all, y_all = line.get_data()
plt.clf()

dd = d2[d2.score > matching_threshold]
my_kde = kdeplot(list(dd.sar_length))
line = my_kde.lines[0]
x_sar, y_sar = line.get_data()
plt.clf()

ais_distribution["all"] = x_all, y_all * len(d2), x_sar, y_sar * len(dd)
# -
# # Figure 3

# +
array = [[1, 1], [2, 3], [2, 4]]  # the "picture" (1 == subplot A, 2 == subplot B, etc.)

fig, axs = pplt.subplots(
    array,
    figwidth=14,
    sharex=False,
    sharey=False,
    grid=False,
    figheight=14,
    spanx=False,
)

###################################
# Plot actual length vs. SAR length
###################################
pplt.rc["abc.size"] = 14
ax = axs[1]  # first plot

## Plot scatter of fishing and non-fishing vessels
alpha = 1

yf = df_m[df_m.is_fishing].sar_length.values
xf = df_m[df_m.is_fishing].gfw_length.values
ax.scatter(xf, yf, alpha=alpha, color=fishing_color, label="Fishing vessels")

ynf = df_m[~df_m.is_fishing].sar_length.values
xnf = df_m[~df_m.is_fishing].gfw_length.values
ax.scatter(xnf, ynf, alpha=alpha, color=nonfishing_color, label="Non-fishing vessels")


## Fit a quantile regression for .33 and .67 regression

y = df_m.sar_length.values
x = df_m.gfw_length.values
quantiles = [0.333, 0.667]
# quantiles = np.linspace(0,100,101)[1:-1]/100
models = [fit_line(x, y, q) for q in quantiles]
models = pd.DataFrame(models, columns=["q", "a", "b", "lb", "ub"])


def length_model(l):
    """SAR length -> AIS length (Q1, Q2, Q3)"""
    return [get_x(a, b, l) for a, b in zip(models.a, models.b)]


x_ = np.linspace(0, x.max(), 50)
quants = {}
ys = []
quantiles_colors = ["#999999", "#4a4a4a"]
for i in range(models.shape[0]):
    q = models.q[i]
    y_ = get_y(models.a[i], models.b[i], x_)
    ys.append(y_)
    quants[q] = y_
    ax.plot(
        x_, y_, linestyle="dashed", label="_nolegend_", linewidth=2, color="dimgray"
    )  # quantiles_colors[i])#'dimgray'


ax.set_ylabel("Length estimated from SAR (m)", fontsize=18)
ax.set_xlabel("Vessel length (m)", fontsize=18)
# ax.set_xticks(range(0, 240, 20))
ax.set_xticks(range(0, 250, 50))
ax.set_xlim(0, 240)
ax.set_ylim(0, 355)
leg = ax.legend(
    frameon=False, prop={"size": 16}, ncols=1, loc="upper left"
)  # bbox_to_anchor=(0.3, 1))
ax.tick_params(axis="both", which="major", labelsize=16)
for lh in leg.legendHandles:
    lh.set_alpha(1)

ax.annotate("Quantile 0.67", xy=(190, 272), fontsize=16, rotation=52, weight="bold")

ax.annotate("Quantile 0.33", xy=(185, 198), fontsize=16, rotation=45.5, weight="bold")

# ax.set_title("Vessel Length vs. Length Estimated by SAR", fontsize=18)


##################################
# Recal as function of length
##################################

ax = axs[0]

x_ = np.linspace(0, 65, 50)


def recall_model(l, max_value=max_detect_rate):
    """Length -> probability of detection
    if recall is greater than the max value give
    it the constant we assigned earlier
    """
    recall = models_.a[0] + models_.b[0] * l
    if np.ndim(recall) > 0:
        recall[recall > max_value] = max_value
    elif recall > max_value:
        recall = max_value
    return recall


x = df_small.gfw_length.values
y = df_small.detected_t.values

x_fish = df_grouped[df_grouped.is_fishing == True].gfw_length.values
y_fish = df_grouped[df_grouped.is_fishing == True].detected_t.values

x_nonfish = df_grouped[(df_grouped.is_fishing == False)].gfw_length.values
y_nonfish = df_grouped[(df_grouped.is_fishing == False)].detected_t.values

# plt.scatter(x,y)


ax.scatter(
    x_fish,
    y_fish,
    color=fishing_color,
    s=30,
    alpha=alpha,
    zorder=2,  # , marker="s"
    label="Fishing vessels",
)

ax.scatter(
    x_nonfish,
    y_nonfish,
    color=nonfishing_color,
    s=30,
    alpha=alpha,
    zorder=2,
    label="Non-fishing vessels",
)


## Median line through the detection rate for vessels < 60m
x = df_small.gfw_length.values
y = df_small.detected_t.values
recall_quantiles = [0.5]
models_ = [fit_line(x, y, q) for q in recall_quantiles]
models_ = pd.DataFrame(models_, columns=["q", "a", "b", "lb", "ub"])
for i in range(models_.shape[0]):
    q = models_.q[i]
    y_ = get_y(models_.a[i], models_.b[i], x_)
    ax.plot(
        x_[(y_ < max_detect_rate) & (y_ > 0)],  # (y_<max_detect_rate)&(y_>0)
        # makes sure that the line
        y_[(y_ < max_detect_rate) & (y_ > 0)],  # doesn't go above max_detect
        linestyle="dashed",  # rate or below 0
        label=f"Median detection rate, vessels < 60m",
        linewidth=2,
        color="darkgray",  # quantiles_colors[0]
    )


## Plot the detection rate for vessels > 60m as a dashed line
ax.plot(
    [50, 250],
    [max_detect_rate, max_detect_rate],
    linestyle="dotted",
    label="Median detection rate, vessels > 60m",
    color=quantiles_colors[1],
)

## Plot average detection rate for vessels binned at 10m
ax.scatter(
    x_bin5[x_bin5 < 60],
    y_bin5[x_bin5 < 60],
    label="Average detection rate, 10m bins",
    marker="x",
    color="black",
    zorder=3,
)


ax.set_xlabel("Vessel length (m)", fontsize=18)
ax.set_ylabel("SAR detection rate", fontsize=18)
ax.set_xticks(range(0, 240, 20))
# plt.ax_xlim(0, 240)
ax.set_ylim(-0.05, 1.05)
leg = ax.legend(frameon=False, ncols=1, prop={"size": 16}, bbox_to_anchor=(0.5, 0.5))
ax.tick_params(axis="both", which="major", labelsize=16)


##### distributions

region = "all"
x_all, y_all, x_sar, y_sar = ais_distribution[region]
all_v = round(y_all.sum() * (x_all[1] - x_all[0]))
sar_v = round(y_sar.sum() * (x_sar[1] - x_sar[0]))

ax = axs[2]

ax.plot(x_all, y_all, label=f"All, {all_v} vessels", color="#003d9e")
ax.plot(
    x_sar, y_sar, label=f"Detected by SAR, {sar_v} vessels", color="#003d9e", alpha=0.4
)
ax.set_ylim(0, 14)
ax.legend(ncols=1, prop={"size": 16}, bbox_to_anchor=(0.3, 0.3))
ax.tick_params(axis="both", which="major", labelsize=16)


ax.set_ylabel("AIS vessels", fontsize=18)
ax.set_xlim(0, 250)

x_all, y_all, x_sar, y_sar = dark_distribution[region]
all_v = round(y_all.sum() * (x_all[1] - x_all[0]))
sar_v = len(df[(df.score < matching_threshold) & (~df.detect_id.isna())])

ax = axs[3]
ax.plot(x_all, y_all, label=f"All, {all_v} $\pm$ 33 vessels", color="black")
ax.plot(
    x_sar, y_sar, label=f"Detected by SAR, {sar_v} vessels", color="black", alpha=0.4
)
ax.set_ylim(0, 14)
ax.legend(ncols=1, prop={"size": 16}, bbox_to_anchor=(0.3, 0.45))
ax.set_xlim(0, 250)
ax.tick_params(axis="both", which="major", labelsize=16)
ax.set_xlabel("Vessel length (m)", fontsize=18)
ax.set_ylabel("Non-broadcasting vessels", fontsize=18)

ax.text(
    80,
    10,
    "Length distribution for vessels detected\n\
by SAR is length estimated by SAR. Length\n\
distribution for all vessels is true length.",
    fontsize=16,
)


###


axs[1].format(aspect=1)

axs.format(abc="a", fontweight="bold")
pplt.rc["abc.size"] = 18
for ax in axs:
    ax.set_anchor("W")
plt.tight_layout()

plt.savefig("figure3.png", dpi=300, bbox_inches="tight")
# -

# # Figure 4

# +
### pplt.rc['axes.titlesize'] = 10
plt.rc("legend", fontsize="9")
plt.rcParams["legend.handlelength"] = 1
plt.rcParams["legend.handleheight"] = 1.125
pplt.rc["abc.size"] = 10


d5 = np.zeros(len(all_sar))
for i in range(len(d5)):
    d5[i] = d[i * 5 : i * 5 + 5].mean()

array = [[1, 2], [3, 4]]  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
fig, axs = pplt.subplots(
    ncols=2,
    nrows=4,
    hratios=(6.5, 4.5, 2.5, 2.3),
    figwidth=5.6,
    sharex=True,
    sharey=True,
    space=0,
    figheight=5.6,
    abc="a",
    abcloc="ul",
    grid=False,
    #                         ylabel='vessels',
    xlabel="length, m",
    # the way I do the left labels is funky. I just make a lot of spaces
    # so that the labels line up in the middle
    leftlabels=(
        "",
        "                              Broadcasting AIS",
        "",
        "                Non-broadcasting",
    ),
)

vessel_ylim = (0, 65)
vessel_ylim2 = (0, 45)
dark_ylim = (0, 23)
vessel_xlim = (0, 200)

legend_x = 0.4

for i, region in enumerate(["indian", "pacific"]):

    d2 = df[~(df.ssvid.isna()) & (df.region == region) & (df.likelihood_in > 0.5)]
    d_ais_only = d2[(d2.score < matching_threshold) | (d2.score.isna())]
    d_ais_sar = d2[d2.score > matching_threshold]

    ax = axs[0 + i]

    d2_sar = d2[~d2.sar_length.isna()]
    ax.hist(
        d2_sar.gfw_length,
        bins=80,
        range=(0, 400),
        color="#245abf",
        label=f"AIS and SAR ({len(d2_sar)})",
    )
    ax.text(
        vessel_xlim[1] * 0.4,
        vessel_ylim[1] * 0.2,
        f"{len(d2)} vessles\nbroadcasting AIS",
    )
    ax.set_ylim(vessel_ylim)
    ax.set_xlim(vessel_xlim)

    ax = axs[2 + i]
    # this is just to include the legend
    ax.hist([-10], color="#245abf", label=f"AIS and SAR ({len(d2_sar)})")
    ax.hist(
        d_ais_only.gfw_length,
        bins=80,
        range=(0, 400),
        color="#90a8d4",
        label=f"AIS only ({len(d2)-len(d2_sar)})",
    )

    ax.set_ylim(vessel_ylim2)

    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(legend_x, 0.95))

    all_sar = results_byregion[region].x
    # these give the disributions
    seen = d5 * all_sar
    not_seen = (1 - d5) * all_sar

    num_seen = len(
        df[
            (df.region == region)
            & (df.score < matching_threshold)
            & (~df.detect_id.isna())
        ]
    )
    num_all_sar = all_sar.sum()

    ax = axs[4 + i]

    best_lengths = []
    for j, v in enumerate(seen):
        v = int(round(v))
        for z in range(v):
            best_lengths.append(j * model_bin_size + model_bin_size / 2)

    ax.hist(
        best_lengths,
        bins=80,
        range=(0, 400),
        color="#4a4a4a",
        label=f" SAR only ({num_seen})",
    )
    ax.set_ylim(dark_ylim)

    ax.format(xlabel="Length (m)")

    ax.text(
        vessel_xlim[1] * 0.3,
        dark_ylim[1] * 0.6,
        f"{num_all_sar:.0f} $\pm$ {.175*num_all_sar:.0f} vessles\n not broadcasting AIS",
    )
    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(legend_x, 0.4))

    ax = axs[6 + i]
    best_lengths = []
    for j, v in enumerate(not_seen):
        v = int(round(v))
        for z in range(v):
            best_lengths.append(j * model_bin_size + model_bin_size / 2)

    ax.hist(
        best_lengths,
        bins=80,
        range=(0, 400),
        color="#9c9c9c",
        label=f"Neither SAR nor AIS\n({num_all_sar-num_seen:.0f} $\pm$ {.175*all_sar.sum():.0f})",
    )
    ax.legend(ncols=1, loc="upper left", bbox_to_anchor=(legend_x, 1))
    ax.set_ylim(dark_ylim)
axs.format(toplabels=("Indian", "Pacific"))
plt.savefig("figure4.png", dpi=300, bbox_inches="tight")
# -
# ## See How Well it Does
#
# Run simulations

# +
likely_dtcts = df[df.likelihood_in > 0.99]
all_indices = np.arange(len(likely_dtcts))
j = 1
if j != 0 and j % 10 == 0:
    print(".", end="", flush=True)
# Vary the number of samples we take so that we can have varying amounts of
# test data. This allows us to evaluate more of the output space.
sample_size = np.random.randint(len(all_indices) // 4, 4 * len(all_indices) // 5)
sample_size

len(all_indices) // 4, 4 * len(all_indices) // 5, len(all_indices)
# -


# +
def run_sims(n_trials):
    array = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    fig, axs = pplt.subplots(array, figwidth=10, share=True, figheight=8)
    plt.rc("legend", fontsize="10")

    pred = []
    act = []
    ds = []

    lmin, lmax = 0, 400
    n_bins = 400
    bins = compute_bins(lmin, lmax, n_bins)
    lengths = compute_lengths(bins)
    likely_dtcts = df[df.likelihood_in > 0.99]
    all_indices = np.arange(len(likely_dtcts))

    for j in range(n_trials):
        if j != 0 and j % 10 == 0:
            print(".", end="", flush=True)
        # Vary the number of samples we take so that we can have varying amounts of
        # test data. This allows us to evaluate more of the output space.
        sample_size = np.random.randint(
            len(all_indices) // 4, 4 * len(all_indices) // 5
        )
        # Bootstrap sample (replace=True) from the train indices to create a model.
        train_indices = np.random.choice(all_indices, size=sample_size, replace=True)

        df_r2 = likely_dtcts.iloc[train_indices]
        # Only SSVIDs not used by the training set can be used for test
        ssvids_for_train = df_r2.ssvid.unique()
        ssvids_for_test = set(likely_dtcts.ssvid) - set(ssvids_for_train)

        # compute L off of only very high confidence matches in the training data
        df_m2 = df_r2[df_r2.match_review == "yes"]
        compute_L = LComputer(df_m2.gfw_length.values, df_m2.sar_length.values)

        # df_r -- for recall as a function of length
        df_large2 = df_r2[(df_r2.gfw_length > 60)]
        max_detect_rate2 = len(df_large2[df_large2.score > matching_threshold]) / len(
            df_large2
        )

        df_small2 = df_r2[(df_r2.gfw_length < 60)]
        df_small2["detected_t"] = df_small2.score.apply(
            lambda x: 1 if x > matching_threshold else 0
        )
        df_small2 = df_small.groupby("ssvid").mean()

        L = compute_L(bins)
        d = compute_d(lengths, max_p=max_detect_rate2, df_small=df_small2)  #

        ds.append(d)

        # Coupling Matrix
        model_bin_size = 5
        assert n_bins % model_bin_size == 0
        # tophat
        C = np.zeros([n_bins, n_bins // model_bin_size])
        for i in range(n_bins):
            C[i, i // model_bin_size] = 1.0 / model_bin_size

        # vessels not in the sample and very likely in the scenes
        df_temp = likely_dtcts[likely_dtcts.ssvid.isin(ssvids_for_test)]
        o = compute_o(df_temp[df_temp.score > matching_threshold], lengths)
        the_result = infer_vessels(o, d, L, bin_coupling_func)

        predicted_vessels = the_result.x.sum()
        actual_vessels = df_temp.likelihood_in.sum()

        pred.append(predicted_vessels)
        act.append(actual_vessels)

        if j < 16:
            ax = axs[j]
            ax.hist(
                df_temp[(df_temp.likelihood_in > 0.5)].gfw_length,
                color="red",
                alpha=0.5,
                bins=50,
                range=(0, 250),
                label="vessels in ais",
            )

            ax.plot(
                [model_bin_size * (1 / 2 + i) for i in range(len(the_result.x))],
                the_result.x,
                "k:",
                linewidth=1,
                label="Expected vessels",
            )
            ax.set_xlim(0, 250)
            ax.set_xlabel("length, m")
            ax.set_ylabel("number of vessels")
            ax.set_title(
                f"Modeled: {the_result.x.sum():.0f}\
          Actual:  {df_temp.likelihood_in.sum():.0f}"
            )

            if j == 15:
                ax.legend(ncols=1)
                plt.savefig(
                    f"sup_model_{matching_threshold}.png", dbpi=300, bbox_inches="tight"
                )
                plt.show()

    return np.array(act), np.array(pred)


# change to True to make your computer do a lot of work
run_simulations = False
n_trials = 10000

if run_simulations:
    act, pred = run_sims(n_trials=n_trials)
    with open(f"samples_{n_trials}_{matching_threshold}.pickle", "wb") as f:
        pickle.dump({"act": act, "pred": pred}, f)
else:
    print("WARNING: Loading canned data since run_simulations is False!")
    with open(f"samples_{n_trials}_{matching_threshold}.pickle", "rb") as f:
        x = pickle.load(f)
    act = x["act"]
    pred = x["pred"]

# +
plt.figure(figsize=(6, 4))
fit = LinearRegression(fit_intercept=False).fit(act[:, None], pred)
x = np.linspace(50, 250)
y = fit.predict(x[:, None])
excess_slope = fit.coef_[0] - 1
# plt.plot(x, y, '--', color="orange")
percent_error = abs(fit.predict(act[:, None]) - pred) / act * 100


plt.title(
    f"Performance of estimating non-broadcasting vessels\n\
mean percent difference +{excess_slope*100:.1f}%, \
mean abs percent error: {percent_error.mean():.1f}%"
)
plt.plot(act, pred, ".", alpha=0.3, markersize=5, markeredgewidth=0)
plt.plot([50, 250], [50, 250], linestyle="--", color="orange")
plt.xlim(40, 250)
plt.ylim(40, 250)
plt.xlabel("actual number of vessels")
plt.ylabel("estimated number of vessels")
plt.savefig("estimation_scatter.png", dpi=300, bbox_inches="tight")

# +
resids = pred - act
resid_width = 50
resid_start, resid_end = 60, 180
locs = np.arange(resid_start, resid_end + 1)
fit_pred = fit.predict(locs[:, None])
w2 = resid_width / 2
cis = []
for lc in locs:
    rs = resids[(act > lc - w2) & (act < lc + w2)]
    cis.append(np.quantile(rs, [0.025, 0.975]))
plt.figure(figsize=(6, 4))
plt.plot(act, pred, ".", markersize=1)
plt.plot(
    locs,
    fit_pred + [x[0] for x in cis],
    "k-",
    linewidth=0.5,
    label="95% confidence bounds",
)
plt.plot(locs, fit_pred + [x[1] for x in cis], "k-", linewidth=0.5)

upper_fit = LinearRegression(fit_intercept=False).fit(
    locs[-50:, None], fit_pred[-50:] + [x[1] for x in cis[-50:]]
)
lower_fit = LinearRegression(fit_intercept=False).fit(
    locs[-50:, None], fit_pred[-50:] + [x[0] for x in cis[-50:]]
)
extended_locs = np.linspace(locs[-1], 250)
plt.plot(
    extended_locs,
    upper_fit.predict(extended_locs[:, None]),
    ":k",
    linewidth=0.5,
    label="extrapolated confidence bounds",
)
plt.plot(extended_locs, lower_fit.predict(extended_locs[:, None]), ":k", linewidth=0.5)
plt.xlim(40, 250)
plt.ylim(40, 250)
plt.xlabel("actual number of vessels")
plt.ylabel("estimated number of vessels")
plt.legend()


# +
def ci_at(x):
    if not (locs[0] < x < locs[-1]):
        raise ValueError("x out of range")
    ndx = np.searchsorted(locs, x)
    return fit_pred[ndx] + cis[ndx]


ndark = 172
print(f"conf int at {ndark}", [round(x) for x in ci_at(ndark)])
# -

plt.hist(pred - act, bins=70)
plt.xlim(-60, 60)
plt.title("distribution of errors on simulations")
plt.ylabel("number of simulations")
plt.xlabel("error (number of vessels)")

# +
from matplotlib.patches import Polygon

resids = pred - act
resid_width = 50
# We want to treat actual vessels as a function of estimated vessels
# to determine the uncertainty in actual vessels

resid_start, resid_end = 15, 210

locs = np.arange(resid_start, resid_end + 1)
fit_pred = fit.predict(locs[:, None])
w2 = resid_width / 2
cis = []
for lc in locs:
    rs = resids[(pred > lc - w2) & (pred < lc + w2)]
    #     cis.append(np.quantile(rs, [0.025, 0.5, 0.975]))
    cis.append(np.quantile(-rs, [0.025, 0.5, 0.975]))


plt.figure(figsize=(6, 4))
plt.plot(act, pred, ".", markersize=1)
# plt.plot(locs, fit_pred + [x[0] for x in cis], "k-", linewidth=0.5,
#          label='95% confidence bounds')
bot = list(zip(locs, fit_pred + [x[0] for x in cis]))
top = list(zip(locs, fit_pred + [x[2] for x in cis]))
plt.plot(
    locs, fit_pred + [x[1] for x in cis], "k-", linewidth=1.0, label="median prediction"
)
bounds = bot + top[::-1]
ax = plt.gca()
ax.add_patch(Polygon(bounds, facecolor=(0, 0, 0, 0.1)))

plt.xlim(60, 180)
plt.ylim(60, 180)
plt.xlabel("actual number of vessels")
plt.ylabel("estimated number of vessels")
plt.legend()


def ci_at(x):
    if not (locs[0] < x < locs[-1]):
        raise ValueError("x out of range")
    ndx = np.searchsorted(locs, x)
    return fit_pred[ndx] + cis[ndx]


print("conf int at 127", [round(x) for x in ci_at(127)])


# -

print("conf int at 172", [round(x) for x in ci_at(171)])

print("conf int at 191", [round(x) for x in ci_at(191)])

# # What fraction of vessels in each region are fishing that are dark?


# +
regions = ["indian", "pacific"]
dark_vessels_ranges = [[140, 171, 200], [16, 19, 22]]

for region, dark_vessels in zip(regions, dark_vessels_ranges):
    print(region)
    d_ = df[(df.region == region) & (df.likelihood_in > 0.5) * (df.gfw_length < 60)]
    ais_vessels = len(d_)
    fishing_vessels = len(d_[d_.is_fishing])
    frac_fishing = fishing_vessels / ais_vessels

    for dark in dark_vessels:
        dark_fishing = dark * frac_fishing
        per_df = dark_fishing / (fishing_vessels + dark_fishing) * 100
        print(
            f"dark vessels: {dark}, dark fishing: {dark_fishing:.0f}, percent dark fishing: {per_df:.1f}%"
        )
