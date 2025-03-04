# %%
import numpy as np
import scipy.io as sio
from statsmodels.stats.weightstats import ttest_ind
import time

import os
cwd = os.getcwd()
print(cwd)

data = sio.loadmat('../data/data_ttest2.mat')
x = data["x"]
y = data["y"]
assert(x.shape == y.shape)

expected = np.squeeze(data["pvalue"])

ncols = x.shape[1]

t0 = time.process_time()
actual = [ttest_ind(x[:,i], y[:,i])[1] for i in range(ncols)]
elapsed_time = time.process_time() - t0
print(elapsed_time)

actual = np.array(actual)

np.testing.assert_almost_equal(expected, actual)

from scipy.stats import sem
from scipy.stats import t

def independent_ttest(data1, data2, alpha):
    # calculate means
    mean1, mean2 = np.mean(data1), np.mean(data2)
    # calculate standard errors
    se1, se2 = sem(data1), sem(data2)
    # standard error on the difference between the samples
    sed = np.sqrt(se1**2.0 + se2**2.0)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1) + len(data2) - 2
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p

t0 = time.process_time()
actual = [independent_ttest(x[:,i], y[:,i], alpha = 0.05)[3] for i in range(ncols)]
elapsed_time = time.process_time() - t0
print(elapsed_time)

actual = np.array(actual)

np.testing.assert_almost_equal(expected, actual)
