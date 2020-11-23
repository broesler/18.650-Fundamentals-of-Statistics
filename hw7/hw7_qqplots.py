#!/usr/bin/env python3
# =============================================================================
#     File: hw7_qqplots.py
#  Created: 2020-10-09 14:09
#   Author: Bernie Roesler
#
"""
  Description: Plot example QQ-plots of various distributions
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.gridspec import GridSpec
from scipy import stats

np.random.seed(565656)

dists = [stats.norm(0, 1),
         stats.uniform(-np.sqrt(3), np.sqrt(3)),
         stats.cauchy(),
         stats.expon(1),
         stats.laplace(np.sqrt(2))]

labels = [r'$\mathcal{N}(0, 1)$',
          r'$\mathcal{U}(-\sqrt{3}, \sqrt{3})$',
          'Cauchy',
          r'$\lambda e^{-\lambda x}, \lambda = 1$',
          r'$\frac{\lambda}{2} e^{-\lambda\|x\|}, \lambda = \sqrt{2}$']

N = 1000

fig = plt.figure(1, clear=True)
gs = GridSpec(nrows=2, ncols=3)

for i, (dist, label) in enumerate(zip(dists, labels)):
    ax = fig.add_subplot(gs[i])

    # Generate data
    X = dist.rvs(N)
    X.sort()
    Fn_inv = X  # => Fn_inv(i/N) == X(i)

    # Calculate normal distribution values
    F_inv = stats.norm(0, 1).ppf([i/N for i in range(1, N+1)])

    ax.plot(F_inv, F_inv, 'k-')
    ax.scatter(F_inv, Fn_inv, s=10, edgecolors='C0', c='None', zorder=99)

    ax.set(title=f"{i+1}: {label}",
           xlabel=r'Theoretical Quantiles $\mathcal{N}(0, 1)$',
           ylabel='Empirical Quantiles',
           xlim=(-3, 3))
    ax.grid('on')

# TODO plot each distribution vs a standard normal for visual comparison

gs.tight_layout(fig)
plt.show()

fig.savefig('./hw7_latex/figures/qqplots.pdf')

# =============================================================================
# =============================================================================
