#!/usr/bin/env python3
# =============================================================================
#     File: hw7_kstest.py
#  Created: 2020-10-10 11:49
#   Author: Bernie Roesler
#
"""
  Description: Create Kolmogorov-Smirnov 2-sample test example.

  X_1, \dots, X_n \sim i.i.d. from distribution with cdf F
  Y_1, \dots, Y_n \sim i.i.d. from distribution with cdf G

  H_0: F = G
  H_1: F \ne G
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.gridspec import GridSpec
from scipy import stats

# Calculate the K-S test statistic
def ks_2samp(X, Y):
    """Compute the Kolmogorov-Smirnov statistic on 2 samples.

    Parameters
    ----------
    X, Y : (N,), (M,) array_like
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can differ.

    Returns
    -------
    statistic : float
        KS statistic.
    pvalue : float
        Two-tailed p-value.

    See Also
    --------
    scipy.stats.ks_2samp
    """
    Tnm = np.max(_ks_2samp(X, Y))

    pvalue = 0.0

    return Tnm, pvalue
    
def _ks_2samp(X, Y):
    """Compute the Kolmogorov-Smirnov statistic on 2 samples.

    Parameters
    ----------
    X, Y : (N,), (M,) array_like
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can differ.

    Returns
    -------
    Tv : (N+1,) ndarray
        Maximum difference in CDFs for each value of X. 

    .. note:: The KS statistic itself is the maximum of these `Tv` values, but
        use this helper function for debugging.

    See Also
    --------
    scipy.stats.ks_2samp
    """
    # TODO 
    #   * store index of maximum difference for plotting?
    n = len(X)
    m = len(Y)
    # Sort copies of the data
    Xs = np.hstack([-np.inf, np.sort(X)])  # pad extra point
    Ys = np.sort(Y)
    # Calculate the maximum difference in the empirical CDFs
    # Tnm = 0
    Tv = np.zeros(n+1)  # extra value at Fn = 0 (Xs -> -infty)
    j = 0
    for i in range(n):
        # Find greatest Ys point j s.t. Ys[j] <= Xs[i] and Xs[i] < Ys[j+1] 
        while j < m and Ys[j] <= Xs[i] and Xs[i] > Ys[j+1]:
            j += 1

        # If Gm(Y) rises before Fn(X), take diff between that point and 0.
        if i == 0 and Ys[j] <= Xs[i]:
            Tv[i] = j/m
            continue

        test_lo = np.abs(i/n - j/m)

        # Find next greatest Ys point k s.t. Ys[k] < X[i+1]
        while j < m and Ys[j] < Xs[i+1]:
            j += 1
        test_hi = np.abs(i/n - j/m)

        Tv[i] = np.max((test_lo, test_hi))

    # Clean up last point
    # any remaining Gm(Y) values get closer to 1
    if j < m and Ys[j] > Xs[-1]:
        Tv[-1] = 1 - j/m

    return Tv


def plot_cdfs(X, Y, fignum=1):
    """Plot empirical CDFs of two samples `X` and `Y`."""
    # TODO generalize to K-samples
    n = len(X)
    m = len(Y)
    Xs = np.sort(X)
    Ys = np.sort(Y)
    Fn = np.arange(1, n+1) / n
    Gm = np.arange(1, m+1) / m

    # Pad range with 0, 1 values for plotting
    xmin = np.min(np.hstack([Xs, Ys])) - 0.5
    xmax = np.max(np.hstack([Xs, Ys])) + 0.5
    Xs_plot = np.hstack([xmin, Xs, xmax])
    Ys_plot = np.hstack([xmin, Ys, xmax])
    Fn_plot = np.hstack([0, Fn, 1])
    Gm_plot = np.hstack([0, Gm, 1])

    fig = plt.figure(fignum, clear=True)
    ax = fig.add_subplot()
    ax.step(Xs_plot, Fn_plot, where='post', label='$F_n(X^{(i)})$')
    ax.step(Ys_plot, Gm_plot, where='post', label='$G_m(Y^{(j)})$')
    ax.scatter(Xs, Fn)
    ax.scatter(Ys, Gm)
    ax.set(xlabel='$X$, $Y$',
        ylabel='$F_n$, $G_m$')
    ax.legend()

    plt.show()


def run_test(n, m, plot=False, fignum=1):
    """Run a single test on the `ks_2samp` function.

    Parameters
    ----------
    n, m : int
        number of X and Y observations, respectively.

    Returns
    -------
    Tv : (N+1,) ndarray
        Array of values of the test statistic.
    """
    # Define the distributions (independent, frozen)
    #   * "unknown" to the statistician performing the test
    Xdist = stats.norm(0, 1)
    Ydist = stats.norm(0, 1)

    # Sample the distributions
    X = Xdist.rvs(n)
    Y = Ydist.rvs(m)

    if plot:
        plot_cdfs(X, Y, fignum)

    return _ks_2samp(X, Y)  # use helper to return entire vector of values

# ----------------------------------------------------------------------------- 
#         Run Tests
# -----------------------------------------------------------------------------
# Define test counts
tests = fails = 0

def should_be(a, b, name=None, verbose=False):
    """Test a condition."""
    global tests, fails
    tests += 1
    try:
        try:
            assert a == b
        except ValueError:
            assert np.allclose(a, b)
        if verbose:
            print(f"[{name}]:\nGot:      {a}\nExpected: {b}")
    except AssertionError as e:
        fails += 1
        print(f"[{name}]:\nGot:      {a}\nExpected: {b}")
        raise e

# ----------------------------------------------------------------------------- 
#         Cases for ks_2samp
# -----------------------------------------------------------------------------
# 1. X rises before Y, L
np.random.seed(565656)  # Y jumps before X
Tv = run_test(n=5, m=2)
should_be(Tv, [0.0, 0.3, 0.1, 0.4, 0.2, 0.0])

# 2. 
np.random.seed(123)  # Y jumps before X, multiple Xs within a Y
Tv = run_test(n=5, m=2)
should_be(Tv, [0.5, 0.3, 0.1, 0.1, 0.3, 0.5])

# 3. 
np.random.seed(123)  # Y jumps before X, multiple Xs within a Y
Tv = run_test(n=2, m=5)
should_be(Tv, [0.4, 0.3, 0.2])

# =============================================================================
# =============================================================================
