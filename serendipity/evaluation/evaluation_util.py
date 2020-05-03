import numpy as np

import scipy.stats as st

FLOAT_MAX = np.finfo(np.float32).max


class EvaluationUtil(object):
    @staticmethod
    def sequence_mrr_score(prediction, target):
        return (1.0 / st.rankdata(prediction)[target]).mean()

    @staticmethod
    def sequence_precision_recall_score(recommendation, target, k=20):
        precision, recall = EvaluationUtil._get_precision_recall(recommendation, target, k)
        return precision, recall

    @staticmethod
    def _get_precision_recall(predictions, targets, k):
        predictions = predictions[:k]
        num_hit = len(set(predictions).intersection(set(targets)))

        return float(num_hit) / len(predictions), float(num_hit) / len(targets)

    @staticmethod
    def get_recommendations(model, test_seq, k=20, exclude_preceding=False):
        recommendations = []
        predictions = []
        sequences = test_seq.sequences[:, :-k]
        for sequence in sequences:
            prediction = -model.predict(sequence)
            if exclude_preceding:
                prediction[sequences] = FLOAT_MAX
            predictions.append(prediction)
            recommendation = prediction.argsort()[:k]
            recommendations.append(recommendation)
        return recommendations, predictions

    @staticmethod
    def append_to_dict(mapper, key, value):
        if key not in mapper:
            mapper[key] = np.array([])

        map_value = mapper.get(key)
        mapper[key] = np.append(map_value, value)
        return mapper

    @staticmethod
    def append_to_dict_of_sets(mapper, key, value):
        if key not in mapper:
            mapper[key] = set()

        map_value = mapper.get(key)
        map_value.update(value)
        return mapper

