#%%
import argparse
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
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import xgboost as xgb
from tqdm import tqdm, trange
from timeit import default_timer as timer
import time
from itertools import combinations

np.random.seed(seed=int(time.time()))

def load_dataset(basename):
    df, labels = utils.load_dataset(basename)
    y = labels[basename].to_numpy()
    return df,labels,y

def setup_cache(basename, df, y, output_path, model):
    cachename = output_path + "cache/" + type(model).__name__ + "_" + basename
    print(f"Cache path {cachename}")
    cache = FeatureCache(cachename, df, y, model, N=100, Fold=5, verbose=0)
    return cache

def select_features(NumSelectFeaturesExp, NumFeatures, TrainingPercentage, basename, df, labels, cache):
    print(f"Selecting features...")
    best_acc = 0
    best_features = None
    pbar = trange(NumSelectFeaturesExp)
    for i in pbar:
        df_train, labels_train, rows = utils.split_train(df, labels, TrainingPercentage)
        df_train = utils.clean_zero_std(df_train, labels_train, basename)
        selected_feature_str = features.obm_select(df_train, labels_train, basename, N=NumFeatures)

    # selected_feature_str = features.random_select(df, n=20)

    # start = timer()
    # X = df.loc[:, selected_feature_str].to_numpy()
    # mean, cil, cih, cirange, std = features.evaluate(X, y, model, N=50, Fold=5)
    # tsec = timer() - start

        start = timer()
        mean, cil, cih, cirange, std = cache.evaluate(selected_feature_str)
        tsec = timer() - start

    # print(f"{i+1}/{NExp} {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f} in {tsec:0.3f}s")

        if mean > best_acc:
            best_acc = mean
            best_features = selected_feature_str
            pbar.set_description(f"Accuracy {best_acc:0.2f}")
    return best_acc,best_features

def get_combinations(best_features):
    list_combinations = list()
    for n in range(1, len(best_features) + 1):
        list_combinations += list(combinations(best_features, n))
    num_comb = len(list_combinations)
    return list_combinations

def evaluate_combinations(cache, list_combinations):
    print(f"Evaluating combinations...")
    best_acc = 0
    best_features = None
    pbar = trange(len(list_combinations))
    for i in pbar: 
        selected_features = list_combinations[i]   
        start = timer()
        mean, cil, cih, cirange, std = cache.evaluate(selected_features)
        tsec = timer() - start

    # print(f"{i+1}/{num_comb} {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f} in {tsec:0.3f}s")

        if mean > best_acc:
            best_acc = mean
            best_features = selected_features
            pbar.set_description(f"Accuracy {best_acc:0.2f}")
    return best_acc,best_features

def evaluate_features(df, y, model, best_features):
    print("Features:")
    for feature in best_features:
        print(f"\t{feature}:")

    X = df.loc[:, best_features].to_numpy()
    mean, cil, cih, cirange, std = features.evaluate(X, y, model, N=150, Fold=5)
    print(f"Best tree: {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f}")

    model = xgb.XGBClassifier(n_jobs=8, max_depth=3, n_estimators=50, learning_rate=.1, tree_method='hist')
    mean, cil, cih, cirange, std = features.evaluate(X, y, model, N=150, Fold=5)
    print(f"Best xbg: {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f}")
    return X

def grid_search_xbg(y, df, best_features):
    print("Parameter optimization for XBG...")
    X = df.loc[:, best_features].to_numpy()
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    xgb_model = xgb.XGBClassifier(n_jobs=8)
    clf = GridSearchCV(xgb_model, {
        'max_depth': [1, 2, 3, 4, 5],
        "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
        'n_estimators': [25, 50, 75, 100],
        'gamma': [0.1, 0.01]
        }, verbose=0, n_jobs=1, cv=kf)
    clf.fit(X, y)
    return clf

def evaluate_xbg(y, df, best_features, clf):
    X = df.loc[:, best_features].to_numpy()
    model = xgb.XGBClassifier(n_jobs=8, 
    max_depth=clf.best_params_['max_depth'], 
    n_estimators=clf.best_params_['n_estimators'], 
    learning_rate=clf.best_params_['learning_rate'], 
    gamma=clf.best_params_['gamma'],
    tree_method='hist')
    mean, cil, cih, cirange, std = features.evaluate(X, y, model, N=150, Fold=5)
    print(f"Best xbg: {mean:0.2f} [{cil:0.2f}, {cih:0.2f}] range {cirange:0.2f} std {std:0.2f}")
    return mean    

def main(args):
    # Config
    NumSelectFeaturesExp = args.num_experiments
    NumFeatures = args.num_features
    TrainingPercentage = args.train

    # Setup
    basename = args.input_label
    df, labels, y = load_dataset(basename)

    output_path = utils.setup_output(args.output)
    print(f"Output path {output_path}")

    model = DecisionTreeClassifier()
    print(f"Using model {type(model).__name__}")

    cache = setup_cache(basename, df, y, output_path, model)

    # Select features
    best_acc, best_features = select_features(NumSelectFeaturesExp, NumFeatures, TrainingPercentage, basename, df, labels, cache)

    filename = f"{output_path}BestSelAcc{best_acc:0.2f}_{utils.file_label()}.txt"
    utils.save_features(filename, best_features)    

    # Evaluate combinations
    list_combinations = get_combinations(best_features)
    best_acc, best_features = evaluate_combinations(cache, list_combinations)
    evaluate_features(df, y, model, best_features)

    filename = f"{output_path}BestCombiAcc{best_acc:0.2f}_{utils.file_label()}.txt"
    utils.save_features(filename, best_features)

    # Evalute final
    clf = grid_search_xbg(y, df, best_features)
    mean = evaluate_xbg(y, df, best_features, clf)

    filename = f"{output_path}BestCombiXBGAcc{mean:0.2f}_{utils.file_label()}.txt"
    utils.save_features(filename, best_features)


def parse_opt():
    parser = argparse.ArgumentParser(description="Run FOBM")

    parser.add_argument("-i", "--input_label", type=str, default = "LA_ID_12M_cont_0pad_PlusBasic",
                        help="label")

    parser.add_argument("-o", "--output", type=str, default="fobmResults",
                        help="output directory")

    parser.add_argument("-n", "--num_experiments", type=int, default=10,
                        help="number of experiments")

    parser.add_argument("-f", "--num_features", type=int, default=5,
                        help="number of features")

    parser.add_argument("-t", "--train", type=float, default=60,
                        help="train percentage")

    opt, unknown = parser.parse_known_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)

