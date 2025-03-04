#%%
import sys,os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import fobm.stats as stats
import fobm.utils as utils
import fobm.features as features
from fobm.cache import FeatureCache

from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tqdm import tqdm, trange
from timeit import default_timer as timer
import time
from itertools import combinations
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
import glob

def eval_features(best_features, basename, df, y):
    X = df.loc[:, best_features].to_numpy()
    nneg = np.count_nonzero(y==0)
    npos = np.count_nonzero(y==1)
    nsub = df.shape[0]
    nfeat = df.shape[1]
    print(f"{basename}: {nfeat} features and {nsub} subjects ({npos} positive, {nneg} negative)")
    print(f"{len(best_features)} selected")
    for i, feature in enumerate(best_features):
        print(f"\t{feature}")

    np.random.seed(seed=0)
    print("Parameter optimization for XBG...")
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    xgb_model = xgb.XGBClassifier(n_jobs=8)
    clf = GridSearchCV(xgb_model,
                   {'max_depth': [1, 2, 3, 4, 5],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
                    'n_estimators': [25, 50, 75, 100]}, verbose=1, n_jobs=1, cv=kf)
    clf.fit(X, y)
    for key in clf.best_params_:
        print(f"\t{key}: {clf.best_params_[key]}")

    model = xgb.XGBClassifier(n_jobs=8, 
        max_depth=clf.best_params_['max_depth'], 
        n_estimators=clf.best_params_['n_estimators'], 
        learning_rate=clf.best_params_['learning_rate'], 
        tree_method='hist')
    mean, cil, cih, cirange, std = features.evaluate(X, y, model, N=150, Fold=5)
    print(f"Best xbg: {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f}")

    features.print_metrics(X, y, model, N=150, Fold=5)
    return mean

basename = 'LA_ID_12M_cont_0pad_PlusBasic'
# basename = 'LA_ID_12M_cont_0pad_PlusBasic_excluded14'
df, labels = utils.load_dataset(basename)
y = labels[basename].to_numpy()

#best_features = utils.load_features("../../fobmResults/BestCombiXBGAcc83.96_2022-12-10_00-33-27_095907.txt")
#best_features = utils.load_features("../../fobmResults/BestCombiXBGAcc83.78_2022-12-10_15-41-08_101332.txt")
#best_features = utils.load_features("../../fobmResults/BestCombiXBGAcc84.17_2022-12-12_23-54-07_159889.txt")


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc68.25_2024-10-07_22-18-27_003777.txt")

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc68.14_2024-10-07_23-13-26_107900.txt")


best_features = utils.load_features("/home/tong/fobmResults/BestCombiAcc64.67_2024-10-10_01-02-27_006682.txt")

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc68.02_2024-10-13_15-22-50_276868.txt")

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc72.05_2024-10-15_21-46-32_416572.txt")



best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc75.32_2024-10-16_07-52-06_527821.txt")

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc75.62_2024-10-16_10-08-27_633501.txt") #200 iteration

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc75.07_2024-10-16_10-33-20_737528.txt") #500 iteration

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc75.94_2024-10-16_11-46-35_842115.txt") #300 iteration

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc77.10_2024-10-16_13-07-03_946189.txt") #400 iteration using paired normalized data, kicking zero item out


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc75.90_2024-10-16_14-25-38_1052576.txt") #400 iteration using paired normalized data, kicking zero item out

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc71.65_2024-10-16_15-43-11_1157465.txt") #400 iteration using paired normalized data, kicking zero item out


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc78.39_2024-10-30_10-54-13_1789566.txt") #200 iteration  60% train using DL 128 FEATURES


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc76.09_2024-10-30_11-52-54_1895475.txt") #200 iteration  70% train using DL 128 FEATURES

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc78.58_2024-10-30_12-13-24_1999599.txt") #10000 iteration  60% train using DL 128 FEATURES

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc79.49_2024-10-30_13-52-47_2106123.txt") #100 iteration  60% train using DL 1024 FEATURES


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc81.85_2024-10-30_16-41-43_2211712.txt") #10000 iteration  60% train using DL 1024 FEATURES

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc81.32_2024-10-30_20-49-12_2325598.txt") #10000 iteration  60% train using DL 1024+ 1099 FEATURES


#best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc81.57_2024-10-30_18-07-17_2324238.txt") #100 iteration  60% train using DL 1024+ 1099 FEATURES

best_features = utils.load_features("/home/tong/fobmResults/BestCombiAcc74.96_2024-10-30_22-36-11_2542104.txt") #100 iteration  60% train using DL 1024+ 1099 FEATURES


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc80.05_2024-10-30_22-43-23_2542104.txt") #200 iteration  60% train using TP 10 DL FROM 1024 + TOP 10 FROM 1099 FEATURES


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc80.63_2024-10-31_10-19-43_2895211.txt") #10000 iteration  60% train using 128 DL FROM 1024 +  1099 FEATURES

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc77.52_2024-10-31_10-19-59_2895267.txt") #10000 iteration  60% train using 128 DL FROM 1024 +  1099 FEATURES


best_features = utils.load_features("/home/tong/fobmResults/BestCombiAcc83.63_2024-11-28_21-03-24_005323.txt") #10000




best_features = utils.load_features("/home/tong/fobmResults/BestCombiAcc83.33_2024-11-28_21-01-39_005322.txt") #10000;for 652 pair 1099 features


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc67.43_2024-11-29_22-34-26_231716.txt") #10000;for 326 pair 1099 features


best_features = utils.load_features("/home/tong/fobmResults/BestCombiAcc83.63_2024-11-30_00-06-25_339240.txt") #10000;for 326 pair 1099 features


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc65.17_2024-12-02_00-05-14_003286.txt") #100;for 326 pair 1065features

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc77.58_2024-12-15_21-27-48_006406.txt") #1000;for 326 pair 1024DLfeatures


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc78.08_2024-12-15_21-23-15_006407.txt") #1000;for 326 pair 1024DLfeatures
best_features = utils.load_features("/home/tong/fobmResults/BestCombiAcc71.18_2024-12-15_23-14-18_213154.txt") #1000;for 326 pair obm1099+1024DLfeatures 


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc73.91_2024-12-15_23-43-40_316905.txt") #1000;for 326 pair 128DLfeatures 

best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc75.79_2024-12-16_00-36-33_423443.txt") #1000;for 326 pair 1099+dl1024DLfeatures 


best_features = utils.load_features("/home/tong/fobmResults/BestCombiXBGAcc76.32_2024-12-17_21-17-49_538271.txt") #1000;for 326 pair 1099+dl128DLfeatures 







# best_features = ["Age_years"]

# best_features = [
# 'Age_years',
# 'ratio_evalue1_slambda_right',
# 'ant_fat-Mode intensity-T12--a180-d1-b10-w3-f5',
# 'ant_fat-Maximum intensity-T12--a45-d3-b20-w1-f3',
# 'ant_fat-Maximum intensity-T12--a45-d3-b20-w1-f1',
# 'ant_fat-Minimum intensity-T12--a360-d1-b10-w1-f3']

eval_features(best_features, basename, df, y)

# best_acc = 0
# for filepath in sorted(glob.iglob('../../fobmResults/BestCombiXBGAcc83*.txt')):
#     print(filepath)
#     best_features = utils.load_features(filepath)
#     acc = eval_features(best_features, basename, df, y)
#     if acc > best_acc:
#         best_acc = acc
#         file_best_features = filepath

# print(f"Best {file_best_features}: {best_acc}")
