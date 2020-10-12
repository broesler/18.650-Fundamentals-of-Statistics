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

# np.random.seed(565656)  # Y jumps before X
np.random.seed(123)  # X jumps before Y

# Create two artificial data sets on which to perform a K-S test
n = 2  # number of X values
m = 5  # number of Y values

# ----------------------------------------------------------------------------- 
#         Cases for ks_2samp
# -----------------------------------------------------------------------------
# 1. X rises before Y, L
# np.random.seed(565656)  # Y jumps before X
# n = 5  # number of X values
# m = 2  # number of Y values
# should_be(Tv, [0.3, 0.1, 0.4, 0.2, 0.0])
# -----
# 2. 
# np.random.seed(123)  # Y jumps before X, multiple Xs within a Y
# n = 5  # number of X values
# m = 2  # number of Y values
# should_be(Tv, [0.3, 0.1, 0.1, 0.3, 0.5])
# -----
# 3. 
# np.random.seed(123)  # Y jumps before X, multiple Xs within a Y
# n = 2  # number of X values
# m = 5  # number of Y values
# should_be(Tv, [0.3, 0.2])
# -----

# Define the distributions (independent, frozen)
#   * "unknown" to the statistician performing the test
Xdist = stats.norm(0, 1)
Ydist = stats.norm(0, 1)

# Sample the distributions
X = Xdist.rvs(n)
Y = Ydist.rvs(m)

# Plot the empirical CDFs Fn, Gm (values at jumps)
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

fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
ax.step(Xs_plot, Fn_plot, where='post', label='$F_n(X^{(i)})$')
ax.step(Ys_plot, Gm_plot, where='post', label='$G_m(Y^{(j)})$')
ax.scatter(Xs, Fn)
ax.scatter(Ys, Gm)
ax.set(xlabel='$X$, $Y$',
       ylabel='$F_n$, $G_m$')
ax.legend()

# Calculate the K-S test statistic
def ks_2samp(X, Y):
    # TODO 
    #   * write docs
    #   * automate second sweep (X, Y), then (Y, X), take maximum.
    n = len(X)
    m = len(Y)
    # Sort copies of the data
    Xs = np.sort(X)
    Ys = np.sort(Y)
    # Calculate the maximum difference in the empirical CDFs
    # Tnm = 0
    Tv = np.zeros(n)
    jdx = np.zeros(n, dtype=int)
    j = 0
    # Find greatest Ys point j s.t. Ys[j] <= Xs[i] and Xs[i] < Ys[j+1] 
    #   set test_lo  # point before Xs[i]
    # Find greatest Ys point k s.t. j < k and Xs[i] <= Ys[k] and Ys[k] < X[i+1]
    #   set test_hi  # point before *next* Xs[i], where Gm[k] is largest
    for i in range(n-1):
        # Find greatest Ys point j s.t. Ys[j] <= Xs[i] and Xs[i] < Ys[j+1] 
        while j < m and Ys[j] <= Xs[i] and Xs[i] > Ys[j+1]:
            j += 1
        test_lo = np.abs((i+1)/n - j/m)

        # Find next greatest Ys point k s.t. Ys[k] < X[i+1]
        while j < m and Ys[j] < Xs[i+1]:
            j += 1
        test_hi = np.abs((i+1)/n - j/m)

        Tv[i] = np.max((test_lo, test_hi))

        jdx[i] = j  # store index of Gm(Y) value for each Fn(X) value

    # Clean up last point
    # any remaining Gm(Y) values get closer to 1
    if j < m and Ys[j] > Xs[-1]:
        Tv[-1] = 1 - j/m

    return Tv, jdx

# TODO move into the function to sweep both ways ~(n+m), O(n+m)
Tv, jdx = ks_2samp(X, Y)
print(Tv)
Tv, jdx = ks_2samp(Y, X)
print(Tv)

# Plot the maximum difference
# for i in range(n-1):
#     ax.plot((Xs[i], Xs[i]), (Fn[i], Gm_plot[jdx[i]]), 'k-')

# alpha = 0.05

plt.show()

# =============================================================================
# =============================================================================
