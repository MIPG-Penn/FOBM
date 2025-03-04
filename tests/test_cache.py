import fobm.cache as cache
import fobm.utils as utils
from sklearn.tree import DecisionTreeClassifier
from fobm.cache import FeatureCache
import numpy as np

def test_get_key():
    cache = FeatureCache("key", None, None, None, None, None)

    features = ["1", "3", "2"]

    actual = cache.get_key(features)
    expected = "1,2,3"
    assert(actual == expected)

    features_src = ["1", "3", "2"]
    assert(len(features) == len(features_src))
    assert(all([a == b for a, b in zip(features, features_src)]))

def test_get_feature_list():
    cache = FeatureCache("key", None, None, None, None, None)

    key = "1,2,3"
    actual = cache.get_feature_list(key)
    expected = ["1", "2", "3"]

    assert(len(actual) == len(expected))
    assert(all([a == b for a, b in zip(actual, expected)]))

def test_cache():
    basename = 'LA_ID_12M_cont_0pad_PlusBasic'
    df, labels = utils.load_dataset(basename)

    y = labels[basename].to_numpy()
    model = DecisionTreeClassifier()

    selected_feature_str = ["ratio_surface_area_left_right", "ratio_evalue1_slambda_right", "Age_years", "evalue1_right", "ant_fat-Minimum intensity-T12--a180-d1-b10-w3-f6"]

    features = FeatureCache(basename, df, y, model, N=100, Fold=5, verbose=1)

    mean1, cil, cih, cirange, std = features.evaluate(selected_feature_str)
    mean2, cil, cih, cirange, std = features.evaluate(selected_feature_str)

    np.testing.assert_almost_equal(mean1, mean2) 