from joblib import Memory
import fobm.features as features

class FeatureCache(object):
    def __init__(self, location, df, y, model, N, Fold, verbose=0):
        mem = Memory(location)
        self.cache_evaluate = mem.cache(self.cache_evaluate, verbose=verbose)

        self.df = df
        self.y = y
        self.model = model
        self.N = N
        self.Fold = Fold

    def get_key(self, feature_list):
        feature_list_sorted = sorted(feature_list)
        key = ','.join(feature_list_sorted)
        return key

    def get_feature_list(self, key):
        key_list = key.split(",")
        return key_list[0:]        

    def evaluate(self, selected_feature_str):
        key = self.get_key(selected_feature_str)
        return self.cache_evaluate(key)

    def cache_evaluate(self, key):
        selected_feature_str = self.get_feature_list(key)
        X = self.df.loc[:, selected_feature_str].to_numpy()
        return features.evaluate(X, self.y, self.model, self.N, self.Fold)  