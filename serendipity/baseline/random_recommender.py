import numpy as np


class RandomItemRecommender(object):

    def __init__(self):

        self._num_items = None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

    def fit(self, interactions):
        self._initialize(interactions)

    def predict(self, user_ids, item_ids=None):
        predictions = np.random.rand(self._num_items)
        predictions[self._num_items - 1] = 0
        return predictions
