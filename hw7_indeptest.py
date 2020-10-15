#!/usr/bin/env python3
# =============================================================================
#     File: hw7_indeptest.py
#  Created: 2020-10-14 17:06
#   Author: Bernie Roesler
#
"""
  Description: Simulate 2-sample independence test
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from matplotlib.gridspec import GridSpec

sns.set_style('whitegrid')
rng = np.random.default_rng(seed=565656)

n = 100
dist = stats.norm(0, 1)  # distribution from which to sample 

X = dist.rvs(n)
Y = dist.rvs(n)

# Get the ranks of the r.v.s
R = X.argsort() + 1
Q = Y.argsort() + 1

# Tn = Pearson's correlation coefficient of ranks
#    = Spearman's rank correlation coefficient (aka "Spearman's rho")
# One way of calculating
Rn_bar = R.mean()
Qn_bar = Q.mean()
Rd = R - Rn_bar
Qd = Q - Qn_bar
Tn_1 = Rd.dot(Qd) / np.sqrt(Rd.dot(Rd) * Qd.dot(Qd))

# Simplified expression
Tn = 12 / (n*(n**2 - 1)) * R.dot(Q) - 3*(n+1)/(n-1)

# Alternate calculation (assumes no ties)
rho_p = 1 - 6 * (R - Q).dot(R - Q) / (n*(n**2 - 1))

np.testing.assert_allclose(Tn, Tn_1)
np.testing.assert_allclose(Tn, rho_p)

# Estimate q_alpha of Sn
M = 10000
alpha = 0.05

Sv = np.zeros(M)
for i in range(M):
    Rp = rng.permutation(R)
    Qp = rng.permutation(Q)
    Sv[i] = 12 / (n*(n**2 - 1)) * Rp.dot(Qp) - 3*(n+1)/(n-1)

Sv.sort()

# Calculate statistics
q_hat   = Sv[np.floor(M*(1 - alpha/2)).astype(int)]
d_alpha = np.abs(Tn) > q_hat
pvalue  = np.sum(Sv > Tn) / M

# Normalize Sv (under H0, Sv -> 0 as n -> infty)
Sv_norm = np.sqrt(n) * Sv

# Plot the distribution of Sn vs. N(0, sigma**2)
rv = stats.norm(0, 1)  
x = np.linspace(rv.ppf(0.001), rv.ppf(1 - 0.001), 1000)

fig = plt.figure(1, clear=True, figsize=(12, 6))
gs = GridSpec(nrows=1, ncols=2)
ax = fig.add_subplot(gs[0])
sns.histplot(Sv_norm, stat='density', kde=True, ax=ax, label='KDE $S_n$')
ax.plot(x, rv.pdf(x), 'k-', label=r'$\mathcal{N}(0,1)$')
ax.set(xlabel=r'$x$')
ax.legend(loc='upper left')

ax = fig.add_subplot(gs[1])
sns.ecdfplot(Sv_norm, ax=ax, zorder=9, label=r'ECDF $S_n$')
ax.plot(x, rv.cdf(x), 'k-', label=r'$\Phi(x)$')
ax.set(xlabel=r'$x$')
ax.legend(loc='upper left')

gs.tight_layout(fig)

plt.show()
# =============================================================================
# =============================================================================
