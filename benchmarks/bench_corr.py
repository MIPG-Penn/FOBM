#%%
import sys,os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.utils.benchmark as benchmark

import fobm.stats as stats
import fobm.utils as utils

# Compare takes a list of measurements which we'll save in results.
results = []

basename = 'LA_ID_12M_cont_0pad_PlusBasic'

df, labels = utils.load_dataset(basename)
df = utils.remove_features_zero_std(df, df)
df = utils.remove_features_zero_std(df, df.loc[labels[basename]==1])
df = utils.remove_features_zero_std(df, df.loc[labels[basename]==0])

num_threads = torch.get_num_threads()

results.append(benchmark.Timer(
    stmt='df.corr()',
    setup='import fobm.stats as stats',
    globals={'df': df},
    num_threads=num_threads,
    label="corr benchmark",
    sub_label="pandas",
    description='corr',
).blocked_autorange(min_run_time=1))

results.append(benchmark.Timer(
    stmt='stats.corr_numpy(df)',
    setup='import fobm.stats as stats',
    globals={'df': df},
    num_threads=num_threads,
    label="corr benchmark",
    sub_label="numpy",
    description='corr',
).blocked_autorange(min_run_time=1))

results.append(benchmark.Timer(
    stmt="stats.corr_torch(df, device='cpu')",
    setup='import fobm.stats as stats',
    globals={'df': df},
    num_threads=num_threads,
    label="corr benchmark",
    sub_label="torch (cpu)",
    description='corr',
).blocked_autorange(min_run_time=1))

results.append(benchmark.Timer(
    stmt="stats.corr_torch(df, device='cuda')",
    setup='import fobm.stats as stats',
    globals={'df': df},
    num_threads=num_threads,
    label="corr benchmark",
    sub_label="torch (gpu)",
    description='corr',
).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)

compare.trim_significant_figures()
compare.colorize()
compare.print()