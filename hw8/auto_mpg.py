#!/usr/bin/env python3
# =============================================================================
#     File: auto_mpg.py
#  Created: 2020-10-20 11:27
#   Author: Bernie Roesler
#
"""
  Description: Demo script for auto-mpg data.
  
  Data source: <https://archive.ics.uci.edu/ml/datasets/auto+mpg>
"""
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib.gridspec import GridSpec
from pathlib import Path

# 1. mpg:           continuous
# 2. cylinders:     multi-valued discrete
# 3. displacement:  continuous
# 4. horsepower:    continuous
# 5. weight:        continuous
# 6. acceleration:  continuous
# 7. model year:    multi-valued discrete
# 8. origin:        multi-valued discrete
# 9. car name:      string (unique for each instance)

data_dir = Path('./data/')

dtypes = dict(mpg=float,
              cylinders='category',
              displacement=float,
              horsepower=float,
              weight=float,
              acceleration=float,
              model_year=int,
              origin='category',
              car_name=str)

df = pd.read_csv(data_dir / 'auto-mpg.data', sep='\s+', header=None,
                 names=list(dtypes.keys()), dtype=dtypes, na_values='?').dropna()

# df['log_mpg'] = np.log10(df['mpg'])

# Run linear regressions (multiple forms)
fig = plt.figure(1, clear=True)
ax = fig.add_subplot()
sns.regplot(data=df, x='horsepower', y='mpg', ax=ax)

res = smf.ols('mpg ~ horsepower + displacement + weight', data=df, cov_type='robust').fit()
print('-------------------- mpg ~ horsepower')
print(res.summary())

# # Linear regression on the log of mpg
# fig = plt.figure(2, clear=True)
# ax = fig.add_subplot()
# sns.regplot(data=df, x='horsepower', y='log_mpg', ax=ax)

# res_log = smf.ols('log_mpg ~ horsepower', data=df).fit()
# print('-------------------- log(mpg) ~ horsepower')
# print(res_log.summary())

# Manually compute linear regression
n = df.shape[0]
X = np.c_[np.ones(n), df['horsepower']]  # (n, p)
Y = df['mpg']  # (n,)

# LSE estimator
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y

# np.testing.assert_allclose(beta_hat, res.params)

plt.show()
# =============================================================================
# =============================================================================
