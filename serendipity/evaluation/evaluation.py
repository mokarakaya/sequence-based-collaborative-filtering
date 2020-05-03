import numpy as np
from evaluation.evaluation_util import EvaluationUtil
from serendipity.neighborhood.similarity import similarity_matrix
import sys
import pandas as pd

FLOAT_MAX = np.finfo(np.float32).max




class Evaluation(object):
    def __init__(self, train, test, model, test_seq, k=20, exclude_preceding=True, train_prev=None, dataset=None):
        self.train = train
        self.test = test
        self.model = model
        self.test_seq = test_seq
        self.exclude_preceding = exclude_preceding
        self._sim = similarity_matrix(train.tocsr().T)
        self._sim_prev = None
        self.k = k
        self.dataset = dataset
        if train_prev:
            similarity_matrix(train_prev.tocsr().T)

    def _get_novelty_scores(self):
        df = pd.DataFrame(self.train.item_ids, columns=['item_ids'])
        df = df.groupby('item_ids').item_ids.count()
        return np.insert(np.array(df.values), 0, 0)
        # return np.array(df.values)

    def sequence_evaluation_scores(self, k=20):
        num_users = self.train.num_users
        # novelty_scores = np.array(list(map(lambda data: data.size, self.train.tocsr().T)))
        novelty_scores = self._get_novelty_scores()
        mat = self.test.tocsr()
        recommendation_counts = {}
        count = 0

        mrrs = {}
        precisions = {}
        recalls = {}
        mrrs_at_one = {}
        precisions_at_one = {}
        recalls_at_one = {}
        novelties = {}
        unexpectednesses = {}
        diversities = {}
        aggregate_diversities = {}
        herfindahl_diversities = {}
        entropy_diversities = {}

        sequences = self.test_seq.sequences
        user_ids = self.test_seq.user_ids
        counter = 0
        for sequence, user_id in zip(sequences, user_ids):
            x = np.trim_zeros(sequence)
            for l in range(1, len(x)):

                # random = np.random.rand()
                # if random > 0.1:
                #     continue

                counter += 1
                sequence_input = x[:-l]
                sequence_target = x[-l:]
                sequence_target_at_one = x[-l]


                prediction = -self.model.predict(sequence_input, user_id)
                # import pdb;
                # pdb.set_trace()

                if self.exclude_preceding:
                    prediction[sequence_input] = FLOAT_MAX

                # mrr
                mrr = EvaluationUtil.sequence_mrr_score(prediction, sequence_target)
                mrr_at_one = EvaluationUtil.sequence_mrr_score(prediction, [sequence_target_at_one])

                mrrs = EvaluationUtil.append_to_dict(mrrs, len(sequence_input), mrr)
                mrrs_at_one = EvaluationUtil.append_to_dict(mrrs_at_one, len(sequence_input), mrr_at_one)

                # precision, recall
                recommendation = prediction.argsort()[:k]
                precision, recall = EvaluationUtil.sequence_precision_recall_score(recommendation,
                                                                                   sequence_target,
                                                                                   k)
                precision_at_one, recall_at_one = EvaluationUtil.sequence_precision_recall_score(recommendation,
                                                                                                 [sequence_target_at_one],
                                                                                                 k)
                precisions = EvaluationUtil.append_to_dict(precisions, len(sequence_input), precision)
                recalls = EvaluationUtil.append_to_dict(recalls, len(sequence_input), recall)

                precisions_at_one = EvaluationUtil.append_to_dict(precisions_at_one, len(sequence_input), precision_at_one)
                recalls_at_one = EvaluationUtil.append_to_dict(recalls_at_one, len(sequence_input), recall_at_one)

                # novelty
                # novelty = np.mean(novelty_scores[recommendation])
                novelty = -np.mean(np.log2((novelty_scores[recommendation] + sys.float_info.epsilon) / num_users))
                novelties = EvaluationUtil.append_to_dict(novelties, len(sequence_input), novelty)

                #unexpectedness
                user_interactions = mat[user_id].indices
                distance = []
                for i, first_item in enumerate(recommendation):
                    sum_unexpectedness = np.sum(1 - self._sim[first_item][user_interactions])
                    distance.append(sum_unexpectedness / len(user_interactions))

                unexpectednesses = EvaluationUtil.append_to_dict(unexpectednesses, len(sequence_input), np.mean(distance))

                #diversities
                recommendation_counts = self._update_recommendation_counts(recommendation_counts, recommendation)
                counts = np.array([recommendation_counts[idx] for idx in recommendation])
                count += 1
                entropy = np.array(counts / count)

                entropy_diversity = - np.sum(entropy) * np.log(entropy)
                herfindahl_diversity = 1 - np.sum(entropy * entropy)

                herfindahl_diversities = EvaluationUtil.append_to_dict(herfindahl_diversities, len(sequence_input),
                                                        herfindahl_diversity)
                entropy_diversities = EvaluationUtil.append_to_dict(entropy_diversities, len(sequence_input), entropy_diversity)

                aggregate_diversities = EvaluationUtil.append_to_dict_of_sets(aggregate_diversities,
                                                               len(sequence_input),
                                                               recommendation)
                diversity = self._get_sequence_intra_distance_score(recommendation)
                diversities = EvaluationUtil.append_to_dict(diversities, len(sequence_input), diversity)

        result = {
            "precisions": precisions,
            "recalls": recalls,
            "mrrs": mrrs,
            "diversities": diversities,
            "aggregate_diversities": aggregate_diversities,
            "unexpectednesses": unexpectednesses,
            "novelties": novelties,
            "herfindahl_diversities": herfindahl_diversities,
            "entropy_diversities": entropy_diversities,
            "precisions_at_one": precisions_at_one,
            "recalls_at_one": recalls_at_one,
            "mrrs_at_one": mrrs_at_one
        }
        print('number of recommendations:', counter)
        return result

    def _get_sequence_intra_distance_score(self, recommendation):
        similarities = []

        for i, first_item in enumerate(recommendation):
            for second_item in recommendation[(i + 1):]:
                similarities.append(self._sim[first_item][second_item])
        return 1 - np.array(similarities)

    def _update_recommendation_counts(self, recommendation_counts, recommendation):
        for rec in recommendation:
            if rec in recommendation_counts:
                recommendation_counts[rec] += 1
            else:
                recommendation_counts[rec] = 1.0
        return recommendation_counts
