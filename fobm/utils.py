import pandas as pd
import numpy as np

import os.path
from datetime import datetime

def load_dataset(label):
    data_path = "data/"
    if not os.path.isdir(data_path):
        data_path = "../" + data_path

    features = pd.read_csv(data_path + label + '_features.csv')
    labels = pd.read_csv(data_path + label + '_labels.csv')
    return (features, labels)

def remove_features_zero_std(features, features_std):
    # df_numpy = features.to_numpy()
    # std = df_numpy.std(axis=0)
    # nonzero_std = np.squeeze(np.argwhere(std != 0))
    # return features.iloc[:, nonzero_std]
    return features.loc[:, features_std.std() > 0]

def clean_zero_std(features, labels, classname):
    featuresc = remove_features_zero_std(features, features)
    featuresc = remove_features_zero_std(featuresc, featuresc.loc[labels[classname]==1])
    featuresc = remove_features_zero_std(featuresc, featuresc.loc[labels[classname]==0])    
    return featuresc

def split_train(features, labels, percentage):
    assert(percentage <= 100)
    assert(percentage >= 0)
    assert(features.shape[0] == labels.shape[0])

    indices = np.random.permutation(features.shape[0])
    num_train = round(features.shape[0] * percentage/100)
    rows_train = indices[0:num_train]
    return features.iloc[rows_train, :], labels.iloc[rows_train, :], rows_train

def save_features(filename, features_str):
    with open(filename, 'w') as f:
        for feature in features_str:
            f.write(f"{feature}\n")
            
def load_features(filename):
    with open(filename, 'r') as f:
        features = [line.rstrip('\n') for line in f]
    return features

def setup_output(base = "fobmResults"):
    from pathlib import Path
    output_path = str(Path.home()) + "/" + base + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def file_label():
    now = datetime.now()
    pid = str(os.getpid()).zfill(6)
    file_label = now.strftime("%Y-%m-%d_%H-%M-%S") + "_" + pid    
    return file_label
