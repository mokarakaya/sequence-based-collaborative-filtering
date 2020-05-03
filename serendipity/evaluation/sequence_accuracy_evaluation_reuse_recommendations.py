import numpy as np

import scipy.stats as st
from serendipity.evaluation.sequence_accuracy_evaluation_base import SequenceAccuracyEvaluationBase

FLOAT_MAX = np.finfo(np.float32).max


class SequenceAccuracyEvaluationReuseRecommendations(SequenceAccuracyEvaluationBase):

    def __init__(self, recommendations, predictions):
        self.recommendations = recommendations
        self.predictions = predictions

    def sequence_mrr_score(self, test_seq):
        assert (len(self.predictions) == len(test_seq.sequences)), 'recommendations and test_seq must have same length'
        targets = test_seq.sequences[:, -1:]
        mrrs = []

        for prediction, target in zip(self.predictions, targets):

            mrr = SequenceAccuracyEvaluationBase.sequence_mrr_score(prediction, target)

            mrrs.append(mrr)

        return np.array(mrrs)

    def sequence_precision_recall_score(self, test_seq, k=20):
        assert (len(self.recommendations) == len(test_seq.sequences)), 'recommendations and test_seq must have same length'
        targets = test_seq.sequences[:, -k:]
        precision_recalls = []
        for recommendation, target in zip(self.recommendations, targets):
            precision_recall = SequenceAccuracyEvaluationBase.sequence_precision_recall_score(recommendation, target, k)
            precision_recalls.append(precision_recall)

        precision = np.array(precision_recalls)[:, 0]
        recall = np.array(precision_recalls)[:, 1]
        return precision, recall
