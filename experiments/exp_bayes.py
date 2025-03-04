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

best_features = utils.load_features("../../fobmResults/BestCombiXBGAcc83.96_2022-12-10_00-33-27_095907.txt")
print(best_features)

basename = 'LA_ID_12M_cont_0pad_PlusBasic'
df, labels = utils.load_dataset(basename)
y = labels[basename].to_numpy()
X = df.loc[:, best_features].to_numpy()

kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
xgb_model = xgb.XGBClassifier(n_jobs=8)
clf = GridSearchCV(xgb_model,
                   {'max_depth': [1, 2, 3, 4, 5],
                    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
                    'n_estimators': [25, 50, 75, 100]}, verbose=1, n_jobs=1, cv=kf)
clf.fit(X, y)
# print(clf.best_params_)
# print(clf.best_score_)

model = xgb.XGBClassifier(n_jobs=8, 
    max_depth=clf.best_params_['max_depth'], 
    n_estimators=clf.best_params_['n_estimators'], 
    learning_rate=clf.best_params_['learning_rate'], 
    tree_method='hist')
mean, cil, cih, cirange, std = features.evaluate(X, y, model, N=150, Fold=5)
print(f"Best xbg: {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f}")


#%
model = xgb.XGBClassifier(n_jobs=8, 
    max_depth=6, 
    n_estimators=116, 
    learning_rate=0.0573439349617163,
    gamma = 0.0325992162256199,
    tree_method='hist')
mean, cil, cih, cirange, std = features.evaluate(X, y, model, N=150, Fold=5)
print(f"Best xbg bayes: {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f}")


#%%
from bayes_opt import BayesianOptimization

import xgboost as xgb
from sklearn.metrics import mean_squared_error

dtrain = xgb.DMatrix(X, y)

def bo_tune_xgb(max_depth, gamma, n_estimators, learning_rate):
    params = {'max_depth': int(max_depth),
        'gamma': gamma,
        'n_estimators': int(n_estimators),
        'learning_rate':learning_rate,
        'subsample': 0.8,
        'eta': 0.1,
        'eval_metric': 'rmse'}
    #Cross validating with the specified parameters in 5 folds and 70 iterations
    cv_result = xgb.cv(params, dtrain, num_boost_round=70, nfold=5)
    
    #Return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(bo_tune_xgb, {
    'max_depth': (3, 10),
    'gamma': (0, 1),
    'learning_rate':(0,1),
    'n_estimators':(100,120)
    })

xgb_bo.maximize(n_iter=20, init_points=8, acq='ei')
print('\nbest result:', xgb_bo.max)

params = xgb_bo.max['params']
params['n_estimators'] = int(params['n_estimators'])
params['max_depth'] = int(params['max_depth'])
print(params)

from xgboost import XGBClassifier
classifier2 = XGBClassifier(**params)

mean, cil, cih, cirange, std = features.evaluate(X, y, classifier2, N=150, Fold=5)
print(f"Best xbg: {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f}")

