import numpy as np


class AverageRatingBasedModel(object):

    def __init__(self):

        self.averages = None

    def _initialize(self, interactions):

        item_ids = interactions.item_ids
        ratings = interactions.ratings

        counts = np.bincount(item_ids)
        counts += 1
        sums = np.bincount(item_ids, weights=ratings)
        self.averages = sums / counts

    def fit(self, interactions):
        self._initialize(interactions)

    def predict(self, user_ids, item_ids=None):
        return self.averages
