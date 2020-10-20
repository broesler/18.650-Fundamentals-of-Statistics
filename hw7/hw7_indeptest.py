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
rng = np.random.default_rng()


def spearmanr(X, Y, kind='def'):
    """Calculate Spearman's rank order correlation for two samples.

    Parameters
    ----------
    X, Y : (N, 1) array_like
        Arrays of data over which to calculate the statistic.
    kind : str, optional, {'bydef', 'simple', 'meanvar', 'noties'}
        How to calculate it. Assuming no ties, all kinds are equivalent.

    Returns
    -------
    correlation : float
        The value of the statistic
    pvalue : float \in [0, 1]
        The two-sided p-value for a hypothesis test where H_0 : X \indep Y.
    """
    assert len(X) == len(Y)

    # Get the ranks of the r.v.s
    R = X.argsort() + 1
    Q = Y.argsort() + 1

    n = len(R)

    def _spearmanr_bydef(R, Q):
        """Explicit calculation (by definition)."""
        Rn_bar = R.mean()
        Qn_bar = Q.mean()
        Rd = R - Rn_bar
        Qd = Q - Qn_bar
        return Rd.dot(Qd) / np.sqrt(Rd.dot(Rd) * Qd.dot(Qd))

    func = dict(bydef=_spearmanr_bydef,
                simple=lambda R, Q: 12 / (n*(n**2 - 1)) * R.dot(Q) - 3*(n+1)/(n-1), # Simplified expression
                meanvar=lambda R, Q: 1/R.var() * (1/n * R.dot(Q) - R.mean()**2), # Rewrite simplified expression in terms of mean/variance
                noties=lambda R, Q: 1 - 6 * (R - Q).dot(R - Q) / (n*(n**2 - 1)) # Alternate calculation (assumes no ties)
                )
    
    return func[kind](R, Q)


# ----------------------------------------------------------------------------- 
#         Run an experiment
# -----------------------------------------------------------------------------
n = 100
dist = stats.expon(1)  # distribution from which to sample 

X = dist.rvs(n)
Y = dist.rvs(n)

# Get the ranks of the r.v.s
R = X.argsort() + 1
Q = Y.argsort() + 1

# Tn = Pearson's correlation coefficient of ranks
#    = Spearman's rank correlation coefficient (aka "Spearman's rho")

Tn_1 = spearmanr(X, Y, kind='bydef')
Tn   = spearmanr(X, Y, kind='simple')
Tn_2 = spearmanr(X, Y, kind='meanvar')
Tn_3 = spearmanr(X, Y, kind='noties')

np.testing.assert_allclose(Tn, Tn_1)
np.testing.assert_allclose(Tn, Tn_2)
np.testing.assert_allclose(Tn, Tn_3)

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
pvalue  = np.sum(np.abs(Sv) > np.abs(Tn)) / M  # two-tailed p-value

# Compare to actual scipy values
rho_s, pvalue_s = stats.spearmanr(R, Q)

np.testing.assert_allclose(Tn, rho_s)
# np.testing.assert_allclose(pvalue, pvalue_s, atol=1e-2)

print(f"Tn:      {Tn:.4f}\nq_hat:   {q_hat:.4f}\np-value: {pvalue:.4f}") 
print('scipy.stats.spearmanr values')
print(f"rho_s:   {rho_s:.4f}\np-value: {pvalue_s:.4f}") 

if d_alpha:
    print(f"Reject null, p-value = {100*pvalue:0.2g}%.")
else:
    print(f"Fail to reject null, p-value = {100*pvalue:0.2g}%.")


# ----------------------------------------------------------------------------- 
#         Plot the distribution of Sn vs. N(0, sigma**2)
# -----------------------------------------------------------------------------
# Normalize Sv (under H0, Sv -> 0 as n -> infty)
Sv_norm = np.sqrt(n) * Sv

rv = stats.norm(0, 1)  
x = np.linspace(rv.ppf(0.001), rv.ppf(1 - 0.001), 1000)

fig = plt.figure(1, clear=True, figsize=(12, 6))
fig.suptitle(fr'$(X, Y) \sim \mathrm{{Exp}}(\lambda=1), n = {n}, M = {M}$')
gs = GridSpec(nrows=1, ncols=2)

ax = fig.add_subplot(gs[0])
sns.histplot(Sv_norm, stat='density', kde=True, ax=ax, label='KDE $S_n^M$')
ax.plot(x, rv.pdf(x), 'k-', label=r'$\mathcal{N}(0,1)$')
ax.set(xlabel=r'$x$')
ax.legend(loc='upper left')

ax = fig.add_subplot(gs[1])
sns.ecdfplot(Sv_norm, ax=ax, zorder=9, label=r'ECDF $S_n^M$')
ax.plot(x, rv.cdf(x), 'k-', label=r'$\Phi(x)$')
ax.set(xlabel=r'$x$')
ax.legend(loc='upper left')

gs.tight_layout(fig)

plt.show()
# =============================================================================
# =============================================================================
