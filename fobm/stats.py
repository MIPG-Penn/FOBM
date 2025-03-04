import numpy as np
import torch
from statsmodels.stats.weightstats import ttest_ind
import scipy.stats

def ttest2(x, y):
    """Two-sample t-test
    Parameters
    ----------
    If x and y are specified as matrices, they must have the same number of columns. ttest2 performs a separate t-test along each column and returns a vector of results.

    Returns
    -------
    pvalue
    """
    assert(x.shape[1] == y.shape[1])
    ncols = x.shape[1]
    pvalue = [ttest_ind(x[:,i], y[:,i])[1] for i in range(ncols)]
    return np.array(pvalue)

def class_distance(class1, class2):
    mean1 = class1.mean(axis=0)
    mean2 = class2.mean(axis=0)

    var1 = class1.var(axis=0, ddof=1)
    var2 = class2.var(axis=0, ddof=1)

    nonzero_std = np.squeeze(np.argwhere(var2*var1 == 0))
  #  assert(len(nonzero_std) == 0) # to avoid the crush, tong 
    return np.abs(mean2-mean1)/np.sqrt(var2*var1)

    # a = np.abs(mean2-mean1)
    # b = np.sqrt(var2*var1)
    # c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    # return c


def corr_numpy(features):
    """Compute pairwise correlation of columns
    Parameters
    ----------
    features dataframe

    Returns
    -------
    Correlation matrix.
    """
    features_matrix = features.to_numpy()
    return np.corrcoef(features_matrix, rowvar=False)

def corr_torch(features, device='cuda'):
    features_matrix = torch.tensor(features).T
    features_matrix = features_matrix.to(device)
    return torch.corrcoef(features_matrix).cpu()

def ratio_corr_torch(corrcoef, thresh_corr):
    threshold_mask = torch.logical_and(corrcoef<thresh_corr, corrcoef > -1*thresh_corr)
    sumMatrix = torch.sum(threshold_mask, dim=0)
    return sumMatrix / corrcoef.shape[0]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
