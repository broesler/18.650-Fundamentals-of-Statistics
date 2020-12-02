#!/usr/bin/env python3
# =============================================================================
#     File: seaborn_hist_test.py
#  Created: 2020-12-02 12:52
#   Author: Bernie Roesler
#
"""
  Description:
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.gridspec import GridSpec
from scipy import stats

np.random.seed(565656)

# Sample random variable with known mean/variance
n = 1000
mu = 1
sigma2 = 2
# X = stats.norm(mu, sigma2).rvs(n)
X = stats.expon(1).rvs(n)
Sv_norm = (X - X.mean()) / np.sqrt(X.var())

# Compute standard normal for comparison
rv = stats.norm(0, 1)
x = np.linspace(rv.ppf(0.001), rv.ppf(1 - 0.001), 1000)

fig = plt.figure(1, clear=True)
fig.set_size_inches((8, 4), forward=True)
fig.suptitle(f'n = {n}', fontweight='normal')
gs = GridSpec(nrows=1, ncols=2)

ax = fig.add_subplot(gs[0])
sns.histplot(Sv_norm, stat='density', kde=True, ax=ax, label='KDE')
ax.plot(x, rv.pdf(x), 'k-', label=r'$\mathcal{N}(0,1)$')
ax.set(xlabel=r'$x$')
ax.legend()

ax = fig.add_subplot(gs[1])
sns.ecdfplot(Sv_norm, ax=ax, zorder=9, label=r'ECDF')
ax.plot(x, rv.cdf(x), 'k-', label=r'$\Phi(x)$')
ax.set(xlabel=r'$x$')
ax.legend()

gs.tight_layout(fig)
plt.show()

# =============================================================================
# =============================================================================
