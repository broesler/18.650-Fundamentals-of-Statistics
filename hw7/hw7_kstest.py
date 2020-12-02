#!/usr/bin/env python3
# =============================================================================
#     File: hw7_kstest.py
#  Created: 2020-10-10 11:49
#   Author: Bernie Roesler
#
r"""
  Description: Create Kolmogorov-Smirnov 2-sample test example.

  .. math::
    X_1, \dots, X_n \sim i.i.d. from distribution with cdf F
    Y_1, \dots, Y_n \sim i.i.d. from distribution with cdf G

    H_0: ``F = G''
    H_1: ``F \ne G''
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.gridspec import GridSpec
from scipy import stats


def ks_2samp(X, Y, alpha=0.05):
    r"""Compute the Kolmogorov-Smirnov statistic on 2 samples.

    Parameters
    ----------
    X, Y : (N,), (M,) array_like
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can differ.
    alpha : float \in (0, 1)
        Level of the test.

    Returns
    -------
    statistic : float
        KS statistic at the (1-alpha) level.
    pvalue : float
        Two-tailed p-value.

    See Also
    --------
    scipy.stats.ks_2samp
    """
    M = 1000  # number of samples to take
    n = len(X)
    m = len(Y)

    Tnm = np.max(_ks_2samp(X, Y)[0])  # the test statistic
    Tvs = _sample_Tnm(n, m, M)  # samples of Tnm to estimate quantiles

    # Estimate the (1-alpha) quantile of Tn
    # take value s.t. M*(1-alpha) values are > Tv[q]
    q_hat = Tvs[np.ceil(M*(1 - alpha)).astype(int)]
    pvalue = np.sum(Tvs > Tnm) / M

    return Tnm, pvalue, q_hat


def _sample_Tnm(n, m, M=1000):
    """Simulate M iid copies Tn(1), ..., Tn(M) of the test statistic.

    .. note:: Under the null, Tn is *pivotal*, so it does not depend on the
              underlying distributions of X and Y.

    Parameters
    ----------
    n, m : int
        Number of data points in each sample.
    M : int
        Number of samples to take of the test statistic.

    Returns
    -------
    Tvs : (M,) ndarray
        Sorted samples of the test statistic for computation of quantiles.
    """
    Tv = np.zeros(M)
    # Sample from two arbitrary distributions
    Xs = stats.norm(0, 1).rvs((n, M))
    Ys = stats.norm(0, 1).rvs((m, M))
    for i in range(M):
        Tv[i] = np.max(_ks_2samp(Xs[:, i], Ys[:, i])[0])

    return np.sort(Tv)


# <<begin__ks_2samp>>
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
    """
    n = len(X)
    m = len(Y)
    # Sort copies of the data
    Xs = np.hstack([-np.inf, np.sort(X)])  # pad extra point
    Ys = np.sort(Y)
    # Calculate the maximum difference in the empirical CDFs
    Tv = np.zeros(n+1)  # extra value at Fn = 0 (Xs -> -infty)
    js = np.zeros(n+1, dtype=int)
    j = 0
    for i in range(n+1):
        # Find greatest Ys point j s.t. Ys[j] <= Xs[i] and Xs[i] < Ys[j+1]
        j += _rank(Ys[j:], Xs[i])  # only search remaining values

        test_lo = np.abs(i/n - j/m)
        j_lo = j

        # Find next greatest Ys point k s.t. Ys[k] < X[i+1]
        k = _rank(Ys[j:], Xs[min(i+1, n)]) + j
        test_hi = np.abs(i/n - k/m)
        j_hi = k

        # Take the maximum distance, and corresponding index
        Tv[i] = np.max((test_lo, test_hi))
        js[i] = j_lo if np.argmax((test_lo, test_hi)) == 0 else j_hi

    return Tv, js
# <<end__ks_2samp>>


def _rank(A, k):
    """Return the number of keys in `A` strictly less than `k`."""
    assert all(A == sorted(A))
    lo = 0
    hi = len(A) - 1
    while lo <= hi:
        mid = (hi + lo) // 2
        if k < A[mid]:
            hi = mid - 1
        elif k > A[mid]:
            lo = mid + 1
        else:  # k == A[mid]
            return mid
    return lo


def plot_cdfs(X, Y, fignum=1):
    """Plot empirical CDFs of two samples `X` and `Y`."""
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

    # Compute the test statistic
    Tvs, js = _ks_2samp(X, Y)
    i_max = np.argmax(Tvs)
    j_max = js[i_max]

    # plot difference
    y_ks0 = i_max/n
    y_ks1 = j_max/m

    if y_ks0 > y_ks1:
        x_ks = (Xs_plot[i_max] + Ys_plot[j_max+1]) / 2
    else:
        x_ks = (Xs_plot[i_max+1] + Ys_plot[j_max]) / 2

    fig = plt.figure(fignum, clear=True)
    ax = fig.add_subplot()
    ax.step(Xs_plot, Fn_plot, where='post', c='C0', label=fr"$F_n(X^{{(i)}}), n = {n}$")
    ax.step(Ys_plot, Gm_plot, where='post', c='C3', label=fr"$G_m(Y^{{(j)}}), m = {m}$")

    # Plot the actual test statistic
    ax.annotate(text='', xy=(x_ks, y_ks0), xytext=(x_ks, y_ks1),
                arrowprops=dict(arrowstyle='<->',
                                shrinkA=0.0,  # extend arrow to exact point
                                shrinkB=0.0,
                                lw=2))

    if n < 10 or m < 10:
        ax.scatter(Xs, Fn)
        ax.scatter(Ys, Gm)

    ax.set(xlabel='$X$, $Y$',
           ylabel='$F_n$, $G_m$')
    ax.legend()

    # import ipdb; ipdb.set_trace()
    return fig, ax


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
        plt.show()

    return _ks_2samp(X, Y)[0]  # use helper to return entire vector of values


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
Tv = run_test(n=5, m=2, plot=True)
should_be(Tv, [0.0, 0.3, 0.1, 0.4, 0.2, 0.0])

# 2. Y jumps before X, multiple Xs within a Y
np.random.seed(123)
Tv = run_test(n=5, m=2, plot=True)
should_be(Tv, [0.5, 0.3, 0.1, 0.1, 0.3, 0.5])

# 3. Y jumps before X, multiple Ys within an X
np.random.seed(123)
Tv = run_test(n=2, m=5, plot=True)
should_be(Tv, [0.4, 0.3, 0.2])

# -----------------------------------------------------------------------------
#         Run an actual example
# -----------------------------------------------------------------------------
np.random.seed(565656)
n = 100
m = 70

# Define the distributions (independent, frozen)
#   * "unknown" to the statistician performing the test
Xdist = stats.norm(0, 1)
Ydist = stats.norm(0, 2)

# Sample the distributions
X = Xdist.rvs(n)
Y = Ydist.rvs(m)

Tnm, pvalue, q_hat = ks_2samp(X, Y, alpha=0.05)
print(f"Tnm:     {Tnm:.4f}\nq_hat:   {q_hat:.4f}\np-value: {pvalue:.2e}")

if Tnm > q_hat:
    print(f"Reject null @ {100*pvalue:0.2g}%.")
else:
    print(f"Fail to reject null @ {100*pvalue:0.2g}%.")

# -----------------------------------------------------------------------------
#         Plot the ECDFs
# -----------------------------------------------------------------------------
fig, ax = plot_cdfs(X, Y)
ax.set(title=r'Empirical CDF: $X \sim \mathcal{N}(0,1)$, $Y \sim \mathcal{N}(0, 2)$')
fig.savefig('./hw7_latex/figures/ks_test.pdf')

# -----------------------------------------------------------------------------
#         Plot distribution of the test statistic
# -----------------------------------------------------------------------------
M = 1000
Sv = _sample_Tnm(n, m, M)  # samples of Tnm to estimate quantiles
Sv_norm = (Sv - Sv.mean()) / Sv.std()

# Compute standard normal for comparison
rv = stats.norm(0, 1)
x = np.linspace(rv.ppf(0.001), rv.ppf(1 - 0.001), 1000)

fig = plt.figure(2, clear=True)
fig.set_size_inches((8, 4), forward=True)
fig.suptitle(fr'$X \sim \mathcal{{N}}(0,1)$, $Y \sim \mathcal{{N}}(0, 2)$, n = {n}, m = {m}, M = {M}',
             fontweight='normal')
gs = GridSpec(nrows=1, ncols=2)

ax = fig.add_subplot(gs[0])
sns.histplot(Sv_norm, stat='density', kde=True, ax=ax, label='KDE $S_n^M$')
ax.plot(x, rv.pdf(x), 'k-', label=r'$\mathcal{N}(0,1)$')
ax.set(xlabel=r'$x$')
ax.legend()

ax = fig.add_subplot(gs[1])
# sns.ecdfplot(Sv_norm, ax=ax, zorder=9, label=r'ECDF $S_n^M$')
# ax.plot(x, rv.cdf(x), 'k-', label=r'$\Phi(x)$')
# ax.set(xlabel=r'$x$')
# ax.legend()

# Compute qq plot
# fig = plt.figure(3, clear=True)
# ax = fig.add_subplot()
Fn_inv = Sv_norm
F_inv = stats.norm(0, 1).ppf([i/M for i in range(1, M+1)])
ax.plot(F_inv, F_inv, 'k-')
ax.scatter(F_inv, Fn_inv, s=10, edgecolors='C0', c='None', zorder=99)
ax.set(xlabel=r'Theoretical Quantiles $\mathcal{N}(0, 1)$',
       ylabel='Empirical Quantiles',
       xlim=(-4, 4),
       ylim=(-4, 4.75)
       )

gs.tight_layout(fig)
fig.savefig('./hw7_latex/figures/ks_dist.pdf')

plt.show()
# =============================================================================
# =============================================================================
