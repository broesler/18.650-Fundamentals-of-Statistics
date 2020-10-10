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

np.random.seed(565656)

# Create two artificial data sets on which to perform a K-S test
n = 10  # number of X values
m =  7  # number of Y values

# Define the distributions (independent, frozen)
#   * "unknown" to the statistician performing the test
Xdist = stats.norm(0, 1)
Ydist = stats.norm(0, 1)

# Sample the distributions
X = Xdist.rvs(n)
Y = Ydist.rvs(m)

# Plot the empirical CDFs Fn, Gm (values at jumps)
Xs = X.copy(); Xs.sort()
Ys = Y.copy(); Ys.sort()
Fn = np.arange(1, n+1) / n
Gm = np.arange(1, m+1) / m

# Pad range with 0, 1 values for plotting
xmin = np.min(np.hstack([Xs, Ys])) - 0.5
xmax = np.max(np.hstack([Xs, Ys])) + 0.5
Xs_plot = np.hstack([xmin, Xs[0], Xs, xmax])
Ys_plot = np.hstack([xmin, Ys[0], Ys, xmax])
Fn_plot = np.hstack([0, 0, Fn, 1])
Gm_plot = np.hstack([0, 0, Gm, 1])

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
# Tv = np.zeros(n)
# jdx = np.zeros(n)
# j = 0
# for i in range(n):
#     while (j+1) < m and Ys[j+1] <= Xs[i]:
#         j += 1
#     Tv[i] = np.max((np.abs((i-1)/n - j/m), np.abs(i/n - j/m)))
#     jdx[i] = j  # store index of Gm(Y) value for each Fn(X) value

    # Plot the maximum difference
    # ax.plot([Xs[i], Ys[j]], [Fn[i], Gm[j]])
    # ax.scatter(Xs[i], Fn[i], c='k')

# alpha = 0.05

plt.show()

# =============================================================================
# =============================================================================
