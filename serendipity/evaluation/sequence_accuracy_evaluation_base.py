import numpy as np

import scipy.stats as st

FLOAT_MAX = np.finfo(np.float32).max


class SequenceAccuracyEvaluationBase(object):

    @staticmethod
    def sequence_mrr_score(prediction, target):
        return (1.0 / st.rankdata(prediction)[target]).mean()


    @staticmethod
    def sequence_precision_recall_score(recommendation, target, k=20):
        precision, recall = SequenceAccuracyEvaluationBase._get_precision_recall(recommendation, target, k)
        return precision, recall

    @staticmethod
    def _get_precision_recall(predictions, targets, k):
        predictions = predictions[:k]
        num_hit = len(set(predictions).intersection(set(targets)))

        return float(num_hit) / len(predictions), float(num_hit) / len(targets)

