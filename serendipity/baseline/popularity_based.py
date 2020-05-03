import numpy as np
import sklearn.preprocessing as pp


class PopularityBasedModel(object):

    def __init__(self):

        self.popularity = None

    def _initialize(self, interactions):

        self.popularity = np.bincount(interactions.item_ids)

    def fit(self, interactions):
        self._initialize(interactions)

    def predict(self, user_ids, item_ids=None):
        return pp.minmax_scale(self.popularity, feature_range=(1, 5))
