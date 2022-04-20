# -*- coding: utf-8 -*-
import math
import os
import pickle
import warnings

import cartopy
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcol
import matplotlib.colors as mpcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.ticker import FormatStrFormatter
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
# %matplotlib inline


# import proplot as pplt

import pyseas
import pyseas.cm
import pyseas.contrib as psc
import pyseas.maps
import pyseas.maps as psm
import pyseas.styles
import scipy
import seaborn as sns
import shapely
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import colorbar, colors
from pyseas.contrib import plot_tracks
from scipy.special import erfinv, gammaln
from scipy.stats import binom, gaussian_kde, lognorm, poisson
from shapely import wkt
from sklearn.linear_model import LinearRegression
from itertools import cycle


def get_footprint(detection_table):

    q = """
    select distinct(footprint)
    from `{detection_table}`
    """

    df_footprint = pd.read_gbq(
        q.format(detection_table=detection_table),
        project_id="world-fishing-827",
        dialect="standard",
    )

    df_footprint = gpd.GeoDataFrame(df_footprint)
    df_footprint["geometry"] = df_footprint.footprint.apply(wkt.loads)
    overpasses = shapely.ops.cascaded_union(df_footprint.geometry.values)
    return df_footprint, overpasses


def get_area_shape(sh):
    """pass a shapely polygon or multipolygon, get area in square km"""
    b = sh.bounds
    avg_lat = b[1] / 2 + b[3] / 2
    tot_area = sh.area * 111 * 111 * math.cos(avg_lat * 3.1416 / 180)
    return tot_area


# Create a figure of the footprints in one plot for figure 1
def footprints():
    with pyseas.context(pyseas.styles.dark):
        fig = plt.figure(figsize=(15.5, 5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2)
        bounds1 = mad_footprints[0].total_bounds
        ax1 = pyseas.maps.create_map(
            gs[0],
            projection="regional.indian",
            extent=([bounds1[0] - 1, bounds1[2] + 1, bounds1[1] - 1, bounds1[3] + 1]),
        )
        pyseas.maps.add_land(ax1)
        pyseas.maps.add_eezs(ax1, edgecolor="white", linewidth=0.4)
        pyseas.maps.add_countries(ax1)
        # pyseas.maps.add_scalebar(ax1)
        ax1.add_geometries(
            mad_footprints[0].geometry.values,
            crs=ccrs.PlateCarree(),
            alpha=0.3,
            edgecolor="k",
        )  # for Lat/Lon data.
        ax1.set_extent([bounds1[0] - 1, bounds1[2] + 1, bounds1[1] - 1, bounds1[3] + 1])

        ax1.add_geometries(
            [mad_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            facecolor="none",
            edgecolor="red",
        )  # for Lat/Lon data.

        plt.title("Madagascar", fontsize=14)

        bounds = fp_footprints[0].total_bounds
        ax2 = pyseas.maps.create_map(
            gs[1],
            projection="regional.south_pacific",
            extent=([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1]),
        )
        pyseas.maps.add_land(ax2)
        pyseas.maps.add_eezs(ax2, edgecolor="white", linewidth=0.4)
        pyseas.maps.add_countries(ax2)
        ax2.add_geometries(
            fp_footprints[0].geometry.values,
            crs=ccrs.PlateCarree(),
            alpha=0.3,
            edgecolor="k",
        )  # for Lat/Lon data.
        ax2.set_extent([bounds[0] - 1, bounds[2] + 1, bounds[1] - 1, bounds[3] + 1])

        ax2.add_geometries(
            [fp_footprints[1]],
            crs=ccrs.PlateCarree(),
            alpha=1,
            facecolor="none",
            edgecolor="red",
        )  # for Lat/Lon data.

        plt.title("French Polynesia", fontsize=14)


# for figure 1
def label_axes(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.

    kwargs are collected and passed to `annotate`

    Parameters
    ----------
    fig : Figure
         Figure object to work on

    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.

    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (0.9, 0.9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc, xycoords="axes fraction", **kwargs)


# normal distribution test
# copied the following from
# https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/
def normal_errors_assumption(model, features, label, p_value_thresh=0.05):
    """
    Normality: Assumes that the error terms are normally distributed.
    If they are not,
    nonlinear transformations of variables may solve this.

    This assumption being violated primarily causes issues with
    the confidence intervals
    """
    from statsmodels.stats.diagnostic import normal_ad

    print("Assumption 2: The error terms are normally distributed", "\n")

    # Calculating residuals for the Anderson-Darling test
    df_results = calculate_residuals(model, features, label)

    print("Using the Anderson-Darling test for normal distribution")

    # Performing the test on the residuals
    p_value = normal_ad(df_results["Residuals"])[1]
    print("p-value from the test - below 0.05 generally means non-normal:", p_value)

    # Reporting the normality of the residuals
    if p_value < p_value_thresh:
        print("Residuals are not normally distributed")
    else:
        print("Residuals are normally distributed")

    # Plotting the residuals distribution
    plt.subplots(figsize=(12, 6))
    plt.title("Distribution of Residuals")
    sns.distplot(df_results["Residuals"])
    plt.show()

    print()
    if p_value > p_value_thresh:
        print("Assumption satisfied")
    else:
        print("Assumption not satisfied")
        print()
        print("Confidence intervals will likely be affected")
        print("Try performing nonlinear transformations on variables")


# copied from here: https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/
def calculate_residuals(model, features, label):
    """
    Creates predictions on the features with the model and calculates residuals
    """
    predictions = model.predict(features)
    df_results = pd.DataFrame(
        {
            "Actual": label.flatten(),
            "Predicted": predictions.flatten(),
            "x": features.flatten(),
        }
    )
    df_results["Residuals"] = abs(df_results["Actual"]) - abs(df_results["Predicted"])

    return df_results


def plot_ssvid_scene(fig, ax1, ax2, ax3, ax4, ssvid, scene_id, df):
    plt.rcParams["axes.grid"] = False
    di = df[(df.ssvid == ssvid) & (df.scene_id == scene_id)]

    q = f"""select * from proj_walmart_dark_targets.rasters_single
    where scene_id = '{scene_id}' and ssvid = '{ssvid}' """
    df_r = pd.read_gbq(q)

    the_max_lat = df_r.detect_lat.max()
    the_max_lon = df_r.detect_lon.max()
    the_min_lat = df_r.detect_lat.min()
    the_min_lon = df_r.detect_lon.min()

    start_lon = di.lon1.values[0]
    end_lon = di.lon2.values[0]
    start_lat = di.lat1.values[0]
    end_lat = di.lat2.values[0]
    likely_lon = di.likely_lon.values[0]
    likely_lat = di.likely_lat.values[0]

    ep = 0.01

    the_max_lat = max([the_max_lat, start_lat + ep, end_lat + ep])
    the_max_lon = max([df_r.detect_lon.max(), start_lon + ep, end_lon + ep])
    the_min_lat = min([df_r.detect_lat.min(), start_lat - ep, end_lat - ep])
    the_min_lon = min([df_r.detect_lon.min(), start_lon - ep, end_lon - ep])

    for delta_minutes in sorted(df_r.delta_minutes.unique(), reverse=True):

        d = df_r[df_r.delta_minutes == delta_minutes]

        min_lon = d.detect_lon.min()
        min_lat = d.detect_lat.min()
        max_lon = d.detect_lon.max()
        max_lat = d.detect_lat.max()

        # get the right scale
        if delta_minutes >= 0:
            scale = di.scale1.values[0]
        else:
            scale = di.scale2.values[0]

        pixels_per_degree = int(round(scale * 111))
        pixels_per_degree_lon = int(
            round(scale * 111 * math.cos((min_lat / 2 + max_lat / 2) * 3.1416 / 180))
        )
        num_lons = int(round((max_lon - min_lon) * pixels_per_degree_lon)) + 1
        num_lats = int(round((max_lat - min_lat) * pixels_per_degree)) + 1
        num_lons, num_lats

        grid = np.zeros(shape=(num_lats, num_lons))

        def fill_grid(r):
            y = int(round((r.detect_lat - min_lat) * pixels_per_degree))
            x = int(round((r.detect_lon - min_lon) * pixels_per_degree_lon))
            grid[y][x] += r.probability

        d.apply(fill_grid, axis=1)

        min_value = 1e-7
        max_value = 1
        #     grid[grid<min_value*100]=0
        norm = mpcolors.LogNorm(vmin=min_value, vmax=max_value)

        if delta_minutes >= 0:
            ax = ax1
        else:
            ax = ax2
        ax.imshow(
            np.flipud(grid),
            norm=norm,
            extent=[min_lon, max_lon, min_lat, max_lat],
            interpolation="nearest",
        )

        ax.set_xlim(the_min_lon, the_max_lon)
        ax.set_ylim(the_min_lat, the_max_lat)
        ax.scatter(
            di.detect_lon, di.detect_lat, color="red", s=7, label="Radarsat2 Detection"
        )
        ax.set_facecolor("#440154FF")
        ax.scatter(
            start_lon,
            start_lat,
            s=7,
            color="orange",
            label="Vessel Position Before Detection",
        )
        ax.scatter(
            end_lon,
            end_lat,
            s=7,
            color="purple",
            label="Vessel Position After Detection",
        )
        if delta_minutes >= 0:
            ax.set_title(f"{delta_minutes:.2f} Minutes\nbefore Image", size=19)
        else:
            ax.set_title(f"{-delta_minutes:.2f} Minutes\nafter Image", size=19)
        ax1.set_xlabel("lon", fontsize=17)
        ax1.set_ylabel("lat", fontsize=17)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.tick_params(axis="both", labelsize=16)

        h, l = ax1.get_legend_handles_labels()
        ax4.legend(
            h, l, loc="upper left", markerscale=2.0, ncol=3, frameon=False, fontsize=14
        )
        ax4.set_facecolor("white")

        ax4.set_xticks([])
        ax4.set_yticks([])

        ax4.grid(False)
        ax4.axis("off")

    # include multiplied raster
    q = f"""select * from proj_walmart_dark_targets.rasters_mult where ssvid = '{ssvid}' and scene_id = '{scene_id}' """
    d = pd.read_gbq(q)

    min_lon2 = d.detect_lon.min()
    min_lat2 = d.detect_lat.min()
    max_lon2 = d.detect_lon.max()
    max_lat2 = d.detect_lat.max()

    # scale is the larger
    scale = di.max_scale.values[0]

    pixels_per_degree = int(round(scale * 111))
    pixels_per_degree_lon = int(
        round(scale * 111 * math.cos((min_lat / 2 + max_lat / 2) * 3.1416 / 180))
    )
    num_lons = int(round((max_lon2 - min_lon2) * pixels_per_degree_lon)) + 1
    num_lats = int(round((max_lat2 - min_lat2) * pixels_per_degree)) + 1
    num_lons, num_lats

    grid = np.zeros(shape=(num_lats, num_lons))

    def fill_grid(r):
        y = int(round((r.detect_lat - min_lat2) * pixels_per_degree))
        x = int(round((r.detect_lon - min_lon2) * pixels_per_degree_lon))
        grid[y][x] += r.probability

    d.apply(fill_grid, axis=1)

    ax3.scatter(di.detect_lon, di.detect_lat, color="red", s=7)

    min_value = 1e-7
    max_value = 1

    norm = mpcolors.LogNorm(vmin=min_value, vmax=max_value)
    im = ax3.imshow(
        np.flipud(grid),
        norm=norm,
        extent=[min_lon2, max_lon2, min_lat2, max_lat2],
        interpolation="nearest",
    )

    ax3.set_xlim(the_min_lon, the_max_lon)
    ax3.set_ylim(the_min_lat, the_max_lat)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.grid(False)

    ax3.set_facecolor("#440154FF")
    ax3.set_title("Multiplied Probabilities", size=19)

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(
        im, cax=cax, orientation="vertical", fraction=0.15, aspect=40, pad=0.04
    )

    fontprops = fm.FontProperties(size=14)
    scalebar = AnchoredSizeBar(
        ax3.transData,
        0.09,
        "10 km",
        "lower right",
        pad=0.1,
        color="white",
        frameon=False,
        size_vertical=0,
        fontproperties=fontprops,
    )
    plt.minorticks_off()

    ax3.add_artist(scalebar)


# # Helper Functions for quantile regression and getting results


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


# NOTE: Because this is an inverse function, it will
# return negative values for very small lenghts (<20m),
# so below these lenghts the model is invalid.
def get_x(a, b, y):
    # The inverse function x(y) to be used in the prob
    return (y - a) / b


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


# Dark vessel estimates functions

## Compute the Detection Probility Vector
def compute_bins(lmin, lmax, n):
    return np.linspace(lmin, lmax, n + 1, endpoint=True)


def compute_lengths(bins):
    return 0.5 * (bins[1:] + bins[:-1])


def compute_d(lengths, df_small, min_p=0.01, max_p=1.0):
    x = df_small.gfw_length.values
    y = df_small.detected_t.values
    q, a, b, lb, ub = fit_line(x, y, 0.5)
    return np.clip(get_y(a, b, lengths), min_p, max_p)


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


# Wrapper around optimizer DarkVesselsEstimates
def compute_o(df, lengths, region="none", n_bins=400):
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


# scenes to image longline activity
def get_images(rate, df):
    for index, row in df.iterrows():
        if row.fishing_sum > rate:
            break
    print(f"images to capture {rate} of the fishing: {index}")
    return index


def map_fraction(df, rate, max_value=30):
    index = get_images(rate, df)

    d = df[df.index < index].copy()
    len(d)
    d["one"] = 1
    cells_to_monitor = pyseas.maps.rasters.df2raster(
        d, "lon_index", "lat_index", "one", xyscale=0.25, per_km2=False, origin="lower"
    )

    raster = np.copy(cells_to_monitor)
    plt.rc("text", usetex=False)
    pyseas._reload()
    fig = plt.figure(figsize=(14, 7))
    norm = mpcolors.Normalize(vmin=0, vmax=max_value)
    raster[raster == 0] = np.nan
    with plt.rc_context(pyseas.styles.dark):
        ax, im, cb = psm.plot_raster_w_colorbar(
            raster,
            r"images per cell ",
            cmap="presence",
            norm=norm,
            cbformat="%.0f",
            projection=cartopy.crs.EqualEarth(central_longitude=-157),
            origin="lower",
            loc="bottom",
        )

        psm.add_countries()
        psm.add_eezs()
        ax.set_title(
            f"To image {int(rate*100)}% of Longline Activity in 2020,\
                     {index} Scenes",
            pad=10,
            fontsize=15,
        )
        psm.add_figure_background()
        psm.add_logo(loc="lower right")

        rate_str = str(rate).replace("0.", "").replace
        # plt.savefig(f"images/cells_to_moniotor_longlines_{rate}.png", \
        #  dpi=300, bbox_inches = 'tight')
