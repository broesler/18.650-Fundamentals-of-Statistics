#!/usr/bin/env python3
# =============================================================================
#     File: reg_test.py
#  Created: 2020-10-20 13:10
#   Author: Bernie Roesler
#
"""
  Description: Demo synthetic regression.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib.gridspec import GridSpec
from scipy import stats

np.random.seed(565656)

# Define true parameters
a = 1.0
b = 0.5
beta = np.c_[a, b].T

n = 100        # number of observations
p = beta.size  # number of parameters

# True data points
x = stats.uniform(0, 10).rvs(size=(n, p-1))  # (n,)

err_dist = stats.norm(0, 1)
eps = err_dist.rvs(size=(n,1))

X = np.c_[np.ones(n), x]
Y = X @ beta + eps  # noisy observations

# True line
n_s = 50  # number of "predictions" to make
X_s = np.c_[np.ones(n_s), np.linspace(0, 10, n_s)]
Y_s = X_s @ beta  # no noise

# Estimate the parameters
XTXi = la.inv(X.T @ X)  # only invert once
beta_hat = XTXi @ X.T @ Y

# Make predictions
Y_hat_s = X_s @ beta_hat

# Compute confidence intervals on beta
alpha = 0.05  # 95% interval
sigma_hat2 = 1 / (n - p) * la.norm(Y - X @ beta_hat)  # unbiased estimator
qa = stats.t(n - p).ppf(1 - alpha/2)  # beta_hat is t-distributed
gamma = np.diag(XTXi).reshape(beta_hat.shape)
s = qa * np.sqrt(sigma_hat2 * gamma)
ci_b = np.c_[beta_hat - s, beta_hat + s]  # (p, 2)

# Prediction interval
za = stats.norm(0, 1).ppf(1 - alpha/2)
xi_hat2 = sigma_hat2 * (1 + 1/n * np.sum((x - X_s[:, 1])**2, axis=0) / np.sum((x - x.mean())**2))  # (n_s, 1)
Y_lo = Y_hat_s.reshape(n_s) - za * np.sqrt(xi_hat2)
Y_hi = Y_hat_s.reshape(n_s) + za * np.sqrt(xi_hat2)

residuals = Y - X @ beta_hat

# ----------------------------------------------------------------------------- 
#         Plots
# -----------------------------------------------------------------------------
fig = plt.figure(1, clear=True)
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])
ax.scatter(x, Y, alpha=0.5, label='data')
ax.plot(X_s[:, 1], Y_s, label=f'$y = {a:.4f} + {b:.4f}x$')
ax.plot(X_s[:, 1], Y_hat_s, color='C3', label=f'$\hat{{y}} = {beta_hat[0,0]:.4f} + {beta_hat[1,0]:.4f}x$')
ax.fill_between(X_s[:, 1], Y_lo, Y_hi, color='C3', alpha=0.2)

# Plot possible ranges of beta_hat parameters
# for i in range(2):
#     for j in range(2):
#         ax.plot(X_s[:, 1], X_s @ ci_b[[0, 1], [i, j]], 'k-')

ax.legend()
ax.set(xlabel='$x$',
       ylabel='$y$',
       aspect=1)

ax = fig.add_subplot(gs[1])
ax.scatter(x, residuals, alpha=0.5)
ax.axhline(0, color='k', ls='--')
ax.set(title='Residuals',
       xlabel='$x$',
       ylabel=r'$Y - X\beta$')

gs.tight_layout(fig)

plt.show()
# =============================================================================
# =============================================================================
