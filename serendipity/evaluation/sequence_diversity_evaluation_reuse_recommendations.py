import numpy as np
import scipy.sparse as sp
from serendipity.neighborhood.similarity import similarity_matrix
import sklearn.preprocessing as pp
FLOAT_MAX = np.finfo(np.float32).max
from numba import njit
from numba import jit

class SequenceDiversityEvaluationReuseRecommendations(object):

    def __init__(self, train, test, recommendations, test_seq, train_prev=None):
        self.train = train
        self.test = test
        self.recommendations = recommendations
        self.test_seq = test_seq
        self._sim = similarity_matrix(train.tocsr().T)
        self._sim_prev = None
        if train_prev:
            similarity_matrix(train_prev.tocsr().T)

    def sequence_novelty_score(self):
        novelty_scores = np.array(list(map(lambda data: data.size, self.train.tocsr().T)))
        novelties = []
        for recommendation in self.recommendations:
            novelty = np.mean(novelty_scores[recommendation])
            novelties.append(novelty)
        return novelties

    # @jit
    def sequence_unexpectedness_score(self):

        unexpectednesses = []
        mat = self.test.tocsr()
        user_ids = self.test_seq.user_ids
        for idx, recommendation in enumerate(self.recommendations):
            user_id = user_ids[idx]

            user_interactions = mat[user_id].indices
            distance = []
            for i, first_item in enumerate(recommendation):
                sum_unexpectedness = np.sum(1 - self._sim[first_item][user_interactions])
                distance.append(sum_unexpectedness / len(user_interactions))

            unexpectednesses.append(np.mean(distance))
        return np.array(unexpectednesses)

    def _get_sequence_intra_distance_score(self, recommendation):
        similarities = []

        for i, first_item in enumerate(recommendation):
            for second_item in recommendation[(i + 1):]:
                similarities.append(self._sim[first_item][second_item])
        return 1 - np.array(similarities)

    def sequence_intra_distance_score(self):
        """
        diversity score calculation for sequence models
        """
        diversities = []
        aggregate_diversity = set()

        for recommendation in self.recommendations:
            aggregate_diversity.update(recommendation)
            diversity = self._get_sequence_intra_distance_score(recommendation)
            diversities.append(diversity)

        return diversities, aggregate_diversity
