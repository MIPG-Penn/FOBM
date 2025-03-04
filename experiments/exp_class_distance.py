# %%
%load_ext autoreload
%autoreload 2

import sys,os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import scipy.io as sio

import fobm.stats as stats
import fobm.utils as utils

data = sio.loadmat('../data/data_class_distance.mat')
sel0 = data["sel0"]
sel1 = data["sel1"]

expected = np.squeeze(data["ClassDist"])

actual = stats.calculate_class_distance(sel0, sel1)

np.testing.assert_almost_equal(expected, actual)



