import numpy as np
from serendipity.neighborhood.similarity import similarity_matrix
from serendipity.evaluation.sequence_evaluation_util import append_to_dict, append_to_dict_of_sets
FLOAT_MAX = np.finfo(np.float32).max
import math


class SequenceDiversityEvaluation(object):

    def __init__(self, train, test, model, test_seq, k=20, exclude_preceding=True, train_prev=None):
        self.train = train
        self.test = test
        self.model = model
        self.test_seq = test_seq
        self.exclude_preceding = exclude_preceding
        self._sim = similarity_matrix(train.tocsr().T)
        self._sim_prev = None
        self.k = k
        if train_prev:
            similarity_matrix(train_prev.tocsr().T)

    def sequence_novelty_score(self):
        num_users = self.train.num_users
        novelty_scores = np.array(list(map(lambda data: data.size, self.train.tocsr().T)))
        novelties = {}
        sequences = self.test_seq.sequences
        user_ids = self.test_seq.user_ids
        for sequence, user_id in zip(sequences, user_ids):
            x = np.trim_zeros(sequence)
            for l in range(1, len(x)):
                sequence_input = x[:-l]

                prediction = -self.model.predict(sequence_input, user_id)

                if self.exclude_preceding:
                    prediction[sequences] = FLOAT_MAX

                recommendation = prediction.argsort()[:self.k]
                novelty = -np.mean(np.log2(novelty_scores[recommendation]/num_users))
                novelties = append_to_dict(novelties, len(sequence_input), novelty)
        return novelties

    def sequence_unexpectedness_score(self):

        unexpectednesses = {}
        mat = self.test.tocsr()
        sequences = self.test_seq.sequences
        user_ids = self.test_seq.user_ids
        for sequence, user_id in zip(sequences, user_ids):
            x = np.trim_zeros(sequence)
            for l in range(1, len(x)):
                sequence_input = x[:-l]

                prediction = -self.model.predict(sequence_input, user_id)
                if self.exclude_preceding:
                    prediction[sequences] = FLOAT_MAX
                recommendation = prediction.argsort()[:self.k]

                user_interactions = mat[user_id].indices
                distance = []
                for i, first_item in enumerate(recommendation):
                    sum_unexpectedness = np.sum(1 - self._sim[first_item][user_interactions])
                    distance.append(sum_unexpectedness / len(user_interactions))

                # unexpectednesses.append(np.mean(distance))
                unexpectednesses = append_to_dict(unexpectednesses, len(sequence_input), np.mean(distance))
        return unexpectednesses

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
        diversities = {}
        aggregate_diversities = {}
        herfindahl_diversities = {}
        entropy_diversities = {}

        sequences = self.test_seq.sequences
        user_ids = self.test_seq.user_ids

        recommendation_counts = {}
        count = 0
        for sequence, user_id in zip(sequences, user_ids):
            x = np.trim_zeros(sequence)
            for l in range(1, len(x)):
                sequence_input = x[:-l]

                prediction = -self.model.predict(sequence_input, user_id)

                if self.exclude_preceding:
                    prediction[sequences] = FLOAT_MAX
                recommendation = prediction.argsort()[:self.k]

                recommendation_counts = self._update_recommendation_counts(recommendation_counts, recommendation)
                counts = np.array([recommendation_counts[idx] for idx in recommendation])
                count += 1
                entropy = np.array(counts / count)

                entropy_diversity = - np.sum(entropy * np.log(entropy))
                herfindahl_diversity = 1 - np.sum(entropy * entropy)

                herfindahl_diversities = append_to_dict(herfindahl_diversities, len(sequence_input), herfindahl_diversity)
                entropy_diversities = append_to_dict(entropy_diversities, len(sequence_input),entropy_diversity)

                aggregate_diversities = append_to_dict_of_sets(aggregate_diversities,
                                                               len(sequence_input),
                                                               recommendation)
                diversity = self._get_sequence_intra_distance_score(recommendation)
                diversities = append_to_dict(diversities, len(sequence_input), diversity)


        return diversities, aggregate_diversities, herfindahl_diversities, entropy_diversities

    def _update_recommendation_counts(self, recommendation_counts, recommendation):
        for rec in recommendation:
            if rec in recommendation_counts:
                recommendation_counts[rec] += 1
            else:
                recommendation_counts[rec] = 1.0
        return recommendation_counts