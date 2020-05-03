import numpy as np
import sys
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.preprocessing as pp
from scipy.sparse import bsr_matrix, csr_matrix, coo_matrix


class UserBasedModel(object):

    def __init__(self):

        self._num_users = None
        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None
        self._sim = None
        self._sparse_interactions = None
        self._sparse_interactions_csr = None
        self._interactions_ones = None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

    def _check_input(self, user_ids, item_ids, allow_items_none=False):

        if isinstance(user_ids, int):
            user_id_max = user_ids
        else:
            user_id_max = user_ids.max()

        if user_id_max >= self._num_users:
            raise ValueError('Maximum user id greater '
                             'than number of users in model.')

        if allow_items_none and item_ids is None:
            return

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, interactions):

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)

        self._initialize(interactions)

        self._check_input(user_ids, item_ids)
        self._sparse_interactions = interactions.tocoo()
        self._sparse_interactions_csr = self._sparse_interactions.tocsr()
        self._sim = 1 - pairwise_distances(interactions.tocsr(), metric='cosine')


        # csr_matrix(interactions.c)
    def predict(self, user_ids, item_ids=None):

        self._check_input(user_ids, item_ids, allow_items_none=True)
        return pp.minmax_scale(self._sparse_interactions_csr.T * self._sim[user_ids]) * 5

    def predict_k(self, user_ids, item_ids=None, k=10):
        return 0
