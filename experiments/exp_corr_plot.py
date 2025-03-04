#%%
import sys,os
sys.path.insert(0, os.path.abspath('..'))

%load_ext autoreload
%autoreload 2

#%%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import fobm.stats as stats
import fobm.utils as utils

basename = 'LA_ID_12M_cont_0pad_PlusBasic'

df, labels = utils.load_dataset(basename)
df = utils.remove_features_zero_std(df, df)
df = utils.remove_features_zero_std(df, df.loc[labels[basename]==1])
df = utils.remove_features_zero_std(df, df.loc[labels[basename]==0])

corrcoef = stats.corr_torch(df.values)

cmap = sns.diverging_palette(10, 250, s=75, l=50, sep=100, center='light', as_cmap=True)
pp = sns.clustermap(corrcoef.cpu(), figsize=(13,13), cmap=cmap, cbar_kws={"shrink": .2}, 
    yticklabels=False, xticklabels=False)
_ = plt.setp(pp.ax_heatmap.get_yticklabels(), rotation=0)

# plt.savefig('Corr.png', dpi=600)
plt.show()

