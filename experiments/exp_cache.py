# %%
%load_ext autoreload
%autoreload 2

import sys,os
sys.path.insert(0, os.path.abspath('..'))


#%%
import fobm.utils as utils
from sklearn.tree import DecisionTreeClassifier
from fobm.cache import FeatureCache
from timeit import default_timer as timer
import random

basename = 'LA_ID_12M_cont_0pad_PlusBasic'
df, labels = utils.load_dataset(basename)

y = labels[basename].to_numpy()
model = DecisionTreeClassifier()

selected_feature_str = ["ratio_surface_area_left_right", "ratio_evalue1_slambda_right", "Age_years", "evalue1_right", "ant_fat-Minimum intensity-T12--a180-d1-b10-w3-f6"]

n = 5
selected_feature_str = random.sample(df.columns.tolist(), n)

features = FeatureCache(basename, df, y, model, N=100, Fold=5, verbose=1)

start = timer()
mean1, cil, cih, cirange, std = features.evaluate(selected_feature_str)
tsec1 = timer() - start

start = timer()
mean2, cil, cih, cirange, std = features.evaluate(selected_feature_str)
tsec2 = timer() - start

assert(mean1 == mean2)

print(f"first eval time {tsec1} sec")
print(f"Second eval time {tsec2} sec")

#%%
import json
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

key = features.get_key(selected_feature_str)
print(key)
r.set(key, json.dumps(mean1))

start = timer()
cached = r.get(key)
if cached is not None:
    mean2 = json.loads(cached)
tsec2 = timer() - start

print(mean2)

assert(mean1 == mean2)

print(f"Second eval time with redis {tsec2} sec")
# %%
