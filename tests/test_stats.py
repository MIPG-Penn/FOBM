import numpy as np
import pandas as pd
import torch
import scipy.io as sio

import fobm.stats as stats
import fobm.utils as utils

def test_ttest2():
    data = sio.loadmat('./data/data_ttest2.mat')
    x = data["x"]
    y = data["y"]
    assert(x.shape == y.shape)

    expected = np.squeeze(data["pvalue"])
    actual = stats.ttest2(x, y)

    np.testing.assert_almost_equal(expected, actual)

def test_ttest2_age():
    basename = 'LA_ID_12M_cont_0pad_PlusBasic'

    df, labels = utils.load_dataset(basename)
    df = utils.remove_features_zero_std(df, df)

    df_pos = df.loc[labels[basename]==1]
    df_neg = df.loc[labels[basename]==0]

    print(f"{df_pos.shape[0]} positive, {df_neg.shape[0]} negative")

    df_pos_np = df_pos.to_numpy()
    df_neg_np = df_neg.to_numpy()
    pvalue = stats.ttest2(df_pos_np, df_neg_np)

    idx_age = df.columns.get_loc("Age_years")

    actual = pvalue[idx_age]
    expected = 0.00036751393661112116

    np.testing.assert_almost_equal(expected, actual)

def test_class_distance():
    data = sio.loadmat('./data/data_class_distance.mat')
    sel0 = data["sel0"]
    sel1 = data["sel1"]

    expected = np.squeeze(data["ClassDist"])
    actual = stats.class_distance(sel0, sel1)

    np.testing.assert_almost_equal(expected, actual)

def test_corr():
    basename = 'LA_ID_12M_cont_0pad_PlusBasic'

    df, labels = utils.load_dataset(basename)
    df = df.iloc[:, 1:100]

    expected = df.corr()
    actual = stats.corr_numpy(df)
    np.testing.assert_almost_equal(expected, actual)

def test_corr_torch():
    basename = 'LA_ID_12M_cont_0pad_PlusBasic'

    df, labels = utils.load_dataset(basename)
    df = df.iloc[:, 1:100]

    expected = stats.corr_numpy(df)
    actual = stats.corr_torch(df.values)
    np.testing.assert_almost_equal(expected, actual)

def test_ratio_corr_torch():
    data = sio.loadmat('./data/data_ratio_corr.mat')
    features = data["data"]
    ratio_corr = np.squeeze(data["Ratio_Corr"])
    thresh_corr = torch.tensor(data["thresh_corr"])

    corrcoef = stats.corr_torch(features)

    expected = ratio_corr
    actual = stats.ratio_corr_torch(corrcoef, thresh_corr)
    np.testing.assert_almost_equal(expected, actual)


