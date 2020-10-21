#!/usr/bin/env python3
# =============================================================================
#     File: reg_test.py
#  Created: 2020-10-20 13:10
#   Author: Bernie Roesler
#
"""
  Description: Demo linear regression on synthetic data.
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import seaborn as sns

import statsmodels.api as sm
# import statsmodels.formula.api as smf

from matplotlib.gridspec import GridSpec
from scipy import stats

np.random.seed(565656)

# Define true parameters
beta = np.array([[0.0, 0.0]]).T  # (p, 1)

n = 100        # number of observations
p = beta.size  # number of parameters

# Create synthetic "true" data points
x = stats.uniform(0, 10).rvs(size=(n, p-1))  # (n,)

err_dist = stats.norm(0, 1)
eps = err_dist.rvs(size=(n, 1))

X = np.c_[np.ones_like(x), x]  # (n, p) assume deterministic
Y = X @ beta + eps             # (n, 1) noisy observations

# "true" line
n_s = 50  # number of "predictions" to make
X_s = np.c_[np.ones(n_s), np.linspace(0, 10, n_s)]
Y_s = X_s @ beta  # no noise

# Estimate the parameters
XTXi = np.linalg.inv(X.T @ X)  # only invert once
beta_hat = XTXi @ X.T @ Y

# Make predictions
Y_hat   = X @ beta_hat    # estimate of original data
Y_s_hat = X_s @ beta_hat  # prediction along the line
eps_hat = Y - Y_hat       # residuals (estimate of the actual epsilon)

RSS = np.sum(eps_hat**2)         # residual sum of squared errors
TSS = np.sum((Y - Y.mean())**2)  # total sum of squares
R2 = 1 - RSS/TSS                 # explained variance

# covariance in the data: Cov(X, Y) 
# manually compute
xd = x - x.mean()
yd = Y - Y.mean()
r = (xd.T @ yd / np.sqrt(xd.T @ xd * yd.T @ yd)).squeeze()
# compute using scipy
r_p, pval = stats.pearsonr(x.squeeze(), Y.squeeze())

np.testing.assert_allclose(R2, r**2)
np.testing.assert_allclose(R2, r_p**2)

# Compute confidence intervals on beta_hat
alpha = 0.05                     # 95% interval
beta_dist = stats.t(n - p)       # beta_hat is t-distributed
qa = beta_dist.ppf(1 - alpha/2)  # two-tailed quantile

sigma_hat2 = RSS / (n - p)      # unbiased estimator of error variance
gamma = np.diag(XTXi).reshape(beta_hat.shape)  # variance of data

s = qa * np.sqrt(sigma_hat2 * gamma)
ci_b = np.c_[beta_hat - s, beta_hat + s]  # (p, 2)

# Compute F-statistic 
# Let G = identity, lambda = 0-vector for most basic test
k = p  # using all variables

G = np.eye(beta_hat.size)      # (k, p)
lam = np.zeros_like(beta_hat)  # (k, 1)
gb = G @ beta_hat - lam

Sn = float((gb.T @ np.linalg.inv(G @ XTXi @ G.T) @ gb) / (k*sigma_hat2))
# ISLR, eqn (3.23)
F = ((TSS - RSS) / (p - 1)) / (RSS / (n - p))  # ~ F(k-1, n-p)

F_pvalue = 1 - stats.f(k - 1, n - p).cdf(F)
f_pvalue = 1 - stats.f(k, n - p).cdf(Sn)  # one-tailed since Sn ~ chi-sq

# Compute t-test statistics and pvalues
Tn = (beta_hat / np.sqrt(sigma_hat2 * gamma)).squeeze()
pvalues = 2*(1 - beta_dist.cdf(np.abs(Tn)))

# Prediction interval (*see* Wasserman, Theorem 13.11)
za = stats.norm(0, 1).ppf(1 - alpha/2)
xi_hat2 = sigma_hat2 * (1 + 1/n * np.sum((x - X_s[:, 1])**2, axis=0) / np.sum((x - x.mean())**2))  # (n_s, 1)
Y_lo = Y_s_hat.reshape(n_s) - za * np.sqrt(xi_hat2)
Y_hi = Y_s_hat.reshape(n_s) + za * np.sqrt(xi_hat2)

# Compare vs statsmodels function
res = sm.OLS(Y, X).fit()
# print(res.summary())
np.testing.assert_allclose(beta_hat.squeeze(), res.params)
np.testing.assert_allclose(ci_b, res.conf_int())
# pvalues may be 0.0, so include only absolute tolerance
np.testing.assert_allclose(pvalues, res.pvalues, atol=1e-7)
np.testing.assert_allclose(Tn, res.tvalues)
np.testing.assert_allclose(R2, res.rsquared)
np.testing.assert_allclose(F, res.fvalue)
np.testing.assert_allclose(F_pvalue, res.f_pvalue)

# ----------------------------------------------------------------------------- 
#         Plots
# -----------------------------------------------------------------------------
fig = plt.figure(1, clear=True)
fig.set_size_inches(10, 6, forward=True)
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])
ax.scatter(x, Y, alpha=0.5, label='data')
ax.plot(X_s[:, 1], Y_s, label=f'$y = {beta[0,0]:.4f} + {beta[1,0]:.4f}x$')
ax.plot(X_s[:, 1], Y_s_hat, color='C3', label=fr'$\hat{{y}} = {beta_hat[0,0]:.4f} + {beta_hat[1,0]:.4f}x$')
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
ax.scatter(x, eps_hat, alpha=0.8)
ax.axhline(0, color='k', ls='--')
ax.set(title='Residuals',
       xlabel='$x$',
       ylabel=r'$Y - X\beta$',
       aspect=1)

gs.tight_layout(fig)

plt.show()
# =============================================================================
# =============================================================================
