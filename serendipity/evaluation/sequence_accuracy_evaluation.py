import numpy as np

from serendipity.evaluation.sequence_accuracy_evaluation_base import SequenceAccuracyEvaluationBase
from serendipity.evaluation.sequence_evaluation_util import append_to_dict

FLOAT_MAX = np.finfo(np.float32).max


class SequenceAccuracyEvaluation(SequenceAccuracyEvaluationBase):

    def __init__(self, model, exclude_preceding):
        self.model = model
        self.exclude_preceding = exclude_preceding

    def sequence_mrr_score(self, test_seq):

        mrrs = {}
        sequences = test_seq.sequences
        user_ids = test_seq.user_ids
        for sequence, user_id in zip(sequences, user_ids):
            x = np.trim_zeros(sequence)
            for l in range(1, len(x)):
                sequence_input = x[:-l]
                sequence_target = x[-l:]

                prediction = -self.model.predict(sequence_input, user_id)

                if self.exclude_preceding:
                    prediction[sequence_input] = FLOAT_MAX

                mrr = SequenceAccuracyEvaluationBase.sequence_mrr_score(prediction, sequence_target)

                mrrs = append_to_dict(mrrs, len(sequence_input), mrr)

        return mrrs

    def sequence_precision_recall_score(self, test_seq, k=20):

        precisions = {}
        recalls = {}
        sequences = test_seq.sequences
        user_ids = test_seq.user_ids
        for sequence, user_id in zip(sequences, user_ids):
            x = np.trim_zeros(sequence)
            for l in range(1, len(x)):
                sequence_input = x[:-l]
                sequence_target = x[-l:]
                prediction = -self.model.predict(sequence_input, user_id)
                if self.exclude_preceding:
                    prediction[sequence_input] = FLOAT_MAX
                recommendation = prediction.argsort()[:k]
                precision, recall = SequenceAccuracyEvaluationBase.sequence_precision_recall_score(recommendation,
                                                                                                  sequence_target,
                                                                                                  k)
                precisions = append_to_dict(precisions, len(sequence_input), precision)
                recalls = append_to_dict(recalls, len(sequence_input), recall)

        return precisions, recalls