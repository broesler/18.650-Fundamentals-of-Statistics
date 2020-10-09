#!/usr/bin/env python3
# =============================================================================
#     File: lec6_12.py
#  Created: 2020-08-24 14:55
#   Author: Bernie Roesler
#
"""
  Description: A simple hypothesis test.
"""
# =============================================================================

import numpy as np

data = np.array([[-1.0, -0.8, -2.9,  1.4, 0.3, -0.8,  1.4],
                 [-1.7, -0.1, -0.2,  0.3, 0.3, -0.9, -0.02],
                 [-0.2,  0.6,  1.1, -0.9, 0.1, -1.2,  1.1]])

for i, X in enumerate(data):
    n = X.size
    Xbar_n = X.mean()
    psi = (n**0.5 * np.abs(Xbar_n)) > 0.25  # the test statistic
    result = 'passed' if psi else 'failed'
    print(f"Test {i} {result}.")

# =============================================================================
# =============================================================================
