import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RepeatedKFold, RepeatedStratifiedKFold
import xgboost as xgb
from timeit import default_timer as timer
import fobm.stats as stats

def evaluate(X, y, model, N, Fold):

    # model = xgb.XGBClassifier(n_jobs=8, max_depth=3, n_estimators=50, learning_rate=.1, tree_method='hist')
    # model = DecisionTreeClassifier()

    kf = RepeatedStratifiedKFold(n_splits=Fold, n_repeats=N)
    scores = cross_val_score(model, X, y, cv=kf)

    scores_mat = scores.reshape(N, Fold)
    scores_repetition = scores_mat.mean(axis=1)*100
    mean, cil, cih = stats.mean_confidence_interval(scores_repetition)
    cirange = cih-cil
    std = scores_repetition.std()
    
    return (mean, cil, cih, cirange, std)

def print_metrics(X, y, model, N, Fold):
    kf = RepeatedStratifiedKFold(n_splits=Fold, n_repeats=N)
    accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    balanced_accuracy = cross_val_score(model, X, y, cv=kf, scoring='balanced_accuracy')    
    recall = cross_val_score(model, X, y, cv=kf, scoring='recall')    
    precision = cross_val_score(model, X, y, cv=kf, scoring='precision')    
    f1 = cross_val_score(model, X, y, cv=kf, scoring='precision')    
    roc_auc = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
    sensitivity = recall
    specificity = balanced_accuracy*2 - sensitivity

    print(f"Classification model: {type(model).__name__}")
    print(f"\taccuracy: {np.mean(accuracy):.4f}")
    print(f"\tbalanced_accuracy: {np.mean(balanced_accuracy):.4f}")
    print(f"\trecall: {np.mean(recall):.4f}")
    print(f"\tprecision: {np.mean(precision):.4f}")
    print(f"\tf1: {np.mean(f1):.4f}")
    print(f"\troc_auc: {np.mean(roc_auc):.4f}")
    print(f"\tsensitivity: {np.mean(sensitivity):.4f}")
    print(f"\tspecificity: {np.mean(specificity):.4f}")

def obm_select(df, labels, classname,
    thresh_corr_upper=0.5,
    thresh_corr_lower=0.000001,
    stepThreshCorr=-0.05,
    ratio_number_lower=0.5,
    threshold_pvalue=0.05,
    N = 10,
    corrcoef = None):

    device = 'cuda'

    df_pos = df.loc[labels[classname]==1]
    df_neg = df.loc[labels[classname]==0]

    # print(f"{df.shape[1]} Features and {df.shape[0]} subjects: {df_pos.shape[0]} positive, {df_neg.shape[0]} negative")

    df_pos_np = df_pos.to_numpy()
    df_neg_np = df_neg.to_numpy()

    pvalue = stats.ttest2(df_pos_np, df_neg_np)
    class_distance = stats.class_distance(df_pos_np, df_neg_np)

    if corrcoef is None:
        corrcoef = stats.corr_torch(df.values)
        corrcoef = corrcoef.to(device)

    pvalue = torch.tensor(pvalue).to(device)
    class_distance = torch.tensor(class_distance).to(device)

    feature_size = len(class_distance)

    feature_distance = torch.zeros(feature_size, device=device)
    thresh_corr_eval = torch.arange(start=thresh_corr_upper, end=thresh_corr_lower, step=stepThreshCorr)
    for thresh_corr in thresh_corr_eval:

        ratio_corr = stats.ratio_corr_torch(corrcoef, thresh_corr)

        distance = (class_distance**2 + ratio_corr**2 + (1-thresh_corr)**2).sqrt();
        mask = torch.logical_or(ratio_corr < ratio_number_lower, pvalue > threshold_pvalue)
        distance = distance * ~mask

        feature_distance = torch.max(distance, feature_distance)

    feature_idx_sort = torch.argsort(feature_distance, descending=True)
    selected_feature_idx = feature_idx_sort[0:N].cpu().numpy().tolist()
    selected_feature_str = list(df.columns[selected_feature_idx])
    return selected_feature_str

def random_select(features, n):
    assert(features.shape[1] >= n)

    indices = np.random.permutation(features.shape[1])
    return features.columns[indices[0:n]]

def grid_search_xbg(X, y):
    print("Parameter optimization for XBG...")
    kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
    xgb_model = xgb.XGBClassifier(n_jobs=8)
    clf = GridSearchCV(xgb_model,
        {
            'max_depth': [1, 2, 3, 4],
            "learning_rate": [0.01, 0.1, 0.2],
            'n_estimators': [25, 50, 75],
            'gamma': [0.001, 0.1]
        }, 
        verbose=1, n_jobs=8, cv=kf)
    clf.fit(X, y)
    return clf 