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
import pandas as pd
import seaborn as sns

import statsmodels.api as sm

from matplotlib.gridspec import GridSpec
from scipy import stats

np.random.seed(565656)
sns.set_style('whitegrid')

rng = np.random.default_rng(seed=565656)

# Define true parameters
beta = np.array([[1.0, 0.5]]).T  # (p, 1)

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
x_s = np.linspace(-1, 11, n_s)
X_s = np.c_[np.ones_like(x_s), x_s]
Y_s = X_s @ beta  # no noise

# Estimate the parameters
XTXi = np.linalg.inv(X.T @ X)  # only invert once
beta_hat = XTXi @ X.T @ Y      # == np.linalg.pinv(X) @ Y

# Make predictions
Y_hat   = X @ beta_hat    # estimate of original data
Y_s_hat = X_s @ beta_hat  # prediction along the line
eps_hat = Y - Y_hat       # residuals (estimate of the actual epsilon)

# ----------------------------------------------------------------------------- 
#         Compute Statistics on the Fit
# -----------------------------------------------------------------------------
# Centered data
xc = x - x.mean()
yc = Y - Y.mean()

RSS = float(eps_hat.T @ eps_hat)  # == np.sum(eps_hat**2) # residual sum of squared errors
TSS = float(yc.T @ yc)            # == np.sum((Y - Y.mean())**2)  # total sum of squares
Rsq = 1 - RSS/TSS                  # explained variance

# covariance in the data: Cov(X, Y) 
r = float(xc.T @ yc / np.sqrt(xc.T @ xc * yc.T @ yc))  # manually compute
r_p, pval = stats.pearsonr(x.squeeze(), Y.squeeze())   # compute using scipy

np.testing.assert_allclose(Rsq, r**2)
np.testing.assert_allclose(Rsq, r_p**2)

# Compute confidence intervals on beta_hat
alpha = 0.05                     # 95% interval
beta_dist = stats.t(n - p)       # beta_hat is t-distributed
qa = beta_dist.ppf(1 - alpha/2)  # two-tailed quantile

sigma_hat_sq = RSS / (n - p)      # unbiased estimator of error variance
gamma = np.diag(XTXi).reshape(beta_hat.shape)  # (p, 1) variance of data

var_beta = sigma_hat_sq * gamma           # (p, 1) variance of beta_hat
se_beta = np.sqrt(var_beta)               # (p, 1) standard error of beta_hat
s = qa * se_beta                          # (p, 1) confidence interval quantiles
ci_b = np.c_[beta_hat - s, beta_hat + s]  # (p, 2)

# Compute F-statistic
# Define the null hypothesis H_0: G\beta = \lambda
G = np.eye(beta_hat.size)      # (k, p) restriction coefficients matrix
G[0,0] = 0                     # do not include constant term in restrictions
lam = np.zeros_like(beta_hat)  # (k, 1) restriction values
k = np.linalg.matrix_rank(G)

# Two options:
#   1. take pseudo-inverse in calculation of Sn
#   2. subset X, beta_hat to match dimensions of G, lambda

# Most general form
gb = G @ beta_hat - lam
Sn = float((gb.T @ np.linalg.pinv(G @ XTXi @ G.T) @ gb) / (k*sigma_hat_sq))
f_pvalue = 1 - stats.f(k, n - p).cdf(Sn)  # one-tailed since Sn ~ chi-sq

# ISLR, eqn (3.23)
F = ((TSS - RSS) / (p - 1)) / (RSS / (n - p))  # ~ F(k, n-p)
F_pvalue = 1 - stats.f(k, n - p).cdf(F)

# Compute t-test statistics and pvalues
Tn = (beta_hat / np.sqrt(sigma_hat_sq * gamma)).squeeze()
pvalues = 2*(1 - beta_dist.cdf(np.abs(Tn)))

# Prediction interval (*see* Wasserman, Theorem 13.11)
za = stats.norm(0, 1).ppf(1 - alpha/2)
xd_s = x - x_s
xi_hat2 = sigma_hat_sq * (1 + 1/n * np.sum((x - x_s)**2, axis=0) / np.sum(xc**2))  # (n_s, 1)
Y_lo = Y_s_hat.squeeze() - za * np.sqrt(xi_hat2)
Y_hi = Y_s_hat.squeeze() + za * np.sqrt(xi_hat2)

# Bootstrap a confidence interval for the *function* fit
def bootstrap(X, Y, n_boot=10000):
    """Bootstrap estimates of `beta_hat`."""
    n, p = X.shape
    boot_dist = np.empty((p, n_boot))
    for i in range(n_boot):
        resampler = rng.integers(0, n, n, dtype=np.intp)  # intp is index dtype
        boot_dist[:, [i]] = np.linalg.pinv(X[resampler, :]) @ Y[resampler, :]
    return boot_dist

beta_boots = bootstrap(X, Y)    # (p,   n_boot)
Y_hat_boots = X_s @ beta_boots  # (n_s, n_boot)
err_bands = np.quantile(Y_hat_boots, (alpha/2, 1 - alpha/2), axis=1).T  # (n_s, p)

# TODO 
#   * log-likelihood of the data
#   * AIC
#   * BIC
#   * adjusted Rsq?

# ----------------------------------------------------------------------------- 
#         Compare vs statsmodels function
# -----------------------------------------------------------------------------
res = sm.OLS(Y, X).fit()
# print(res.summary())

np.testing.assert_allclose(Y_hat.squeeze(),    res.fittedvalues)
np.testing.assert_allclose(eps_hat.squeeze(),  res.resid)
np.testing.assert_allclose(beta_hat.squeeze(), res.params)
np.testing.assert_allclose(se_beta.squeeze(),  res.bse)
np.testing.assert_allclose(ci_b,               res.conf_int())
np.testing.assert_allclose(pvalues,            res.pvalues,    atol=1e-7)  # pvalues may be 0.0, so include only absolute tolerance
np.testing.assert_allclose(Tn,                 res.tvalues)
np.testing.assert_allclose(Rsq,                res.rsquared)
np.testing.assert_allclose(Sn,                 res.fvalue)
np.testing.assert_allclose(F,                  res.fvalue)
np.testing.assert_allclose(F_pvalue,           res.f_pvalue,   atol=1e-7)
np.testing.assert_allclose(f_pvalue,           res.f_pvalue,   atol=1e-7)

# Create pandas dataframe for other plots
# df = pd.DataFrame(np.c_[x, Y], columns=['x', 'y'])
# sns.jointplot(x='x', y='y', data=df, kind='reg')

# ----------------------------------------------------------------------------- 
#         Plots
# -----------------------------------------------------------------------------
fig = plt.figure(1, clear=True)
fig.set_size_inches(10, 6, forward=True)
gs = GridSpec(1, 2)
ax = fig.add_subplot(gs[0])
ax.scatter(x, Y, alpha=0.5, label='data')
ax.plot(x_s, Y_s, label=f'$y = {beta[0,0]:.4f} + {beta[1,0]:.4f}x$')
ax.plot(x_s, Y_s_hat, color='C3', label=fr'$\hat{{y}} = {beta_hat[0,0]:.4f} + {beta_hat[1,0]:.4f}x$')

ax.fill_between(x_s, Y_lo, Y_hi, color='C3', alpha=0.1)  # prediction CI
ax.fill_between(x_s, err_bands[:, 0], err_bands[:, 1], color='C3', alpha=0.3)  # curve fit CI

# plot seaborn code:
# yhat = X_s @ np.linalg.pinv(X) @ Y
# ax.scatter(x_s, yhat, marker='*')

# Plot possible ranges of beta_hat parameters
# for i in range(2):
#     for j in range(2):
#         ax.plot(x_s, X_s @ ci_b[[0, 1], [i, j]], 'k-')

ax.legend()
ax.set(xlabel='$x$',
       ylabel='$y$',
       xlim=(-0.5, 10.5))
       #aspect=1)

# Plot residuals
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
