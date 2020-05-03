import csv
import datetime

import numpy as np

from evaluation.evaluation import Evaluation


class ModelEvaluatorDiversity(object):

    def __init__(self,
                 cell_name,
                 dataset,
                 file_path='../../../sequence-based-collaborative-filtering-models/'):
        self.cell_name = cell_name
        self.dataset = dataset
        self.file_path = file_path
        self.results_list = []
        self.evolution_results_list = []

    def evaluate_model(self, model, train, test, test_seq,
                       idx,
                       k=20):

        print('start evaluate_model', datetime.datetime.now())

        evaluation = Evaluation(train, test, model, test_seq, dataset=self.dataset)

        evaluation_scores = evaluation.sequence_evaluation_scores(k=k)

        precisions = evaluation_scores.get('precisions')
        recalls = evaluation_scores.get('recalls')
        mrrs = evaluation_scores.get('mrrs')
        diversities = evaluation_scores.get('diversities')
        aggregate_diversities = evaluation_scores.get('aggregate_diversities')
        herfindahl_diversities = evaluation_scores.get('herfindahl_diversities')
        entropy_diversities = evaluation_scores.get('entropy_diversities')
        unexpectednesses = evaluation_scores.get('unexpectednesses')
        novelties = evaluation_scores.get('novelties')
        precisions_at_one = evaluation_scores.get('precisions_at_one')
        recalls_at_one = evaluation_scores.get('recalls_at_one')
        mrrs_at_one = evaluation_scores.get('mrrs_at_one')

        flatten_precisions = self.flatten(precisions)
        flatten_recalls = self.flatten(recalls)

        precision = np.average(flatten_precisions)
        recall = np.average(flatten_recalls)

        flatten_precisions_at_one = self.flatten(precisions_at_one)
        flatten_recalls_at_one = self.flatten(recalls_at_one)

        precision_at_one = np.average(flatten_precisions_at_one)
        recall_at_one = np.average(flatten_recalls_at_one)

        print('precision, recall', precision, recall)
        print('precision_at_one, recall_at_one', precision_at_one, recall_at_one)

        flatten_mrrs = self.flatten(mrrs)
        mrr = np.average(flatten_mrrs)
        print('mrr', mrr)

        flatten_mrrs_at_one = self.flatten(mrrs_at_one)
        mrr_at_one = np.average(flatten_mrrs_at_one)

        print('mrr_at_one', mrr_at_one)

        flatten_diversities = self.flatten(diversities)
        flatten_aggregate_diversities = self.flatten(aggregate_diversities)

        diversity = np.average(flatten_diversities)

        flatten_herfindahl_diversities = self.flatten(herfindahl_diversities)
        herfindahl_diversity = np.average(flatten_herfindahl_diversities)

        flatten_entropy_diversities = self.flatten(entropy_diversities)
        entropy_diversity = np.average(flatten_entropy_diversities)

        aggregate_diversity = len(set(flatten_aggregate_diversities))

        print('intra_distance_score', diversity, aggregate_diversity, herfindahl_diversity, entropy_diversity)


        flatten_unexpectedness = self.flatten(unexpectednesses)

        unexpectedness = np.average(flatten_unexpectedness)

        print('unexpectedness', unexpectedness)

        flatten_novelties = self.flatten(novelties)

        novelty = np.average(flatten_novelties)

        print('novelty', novelty)

        print('end run', datetime.datetime.now())

        results = {
            'precision': precision,
            'recall': recall,
            'mrr': mrr,
            'diversity': diversity,
            'aggregate_diversity': aggregate_diversity,
            'unexpectedness': unexpectedness,
            'novelty': novelty,
            'herfindahl_diversity': herfindahl_diversity,
            'entropy_diversity': entropy_diversity,
            'precision_at_one': precision_at_one,
            'recall_at_one': recall_at_one,
            'mrr_at_one': mrr_at_one,
        }
        save_path = self.file_path + self.dataset + '/' + self.cell_name + str(idx) + '.search_evaluation_results.csv'

        self.write_to_csv(results, save_path)

        self.results_list.append(results)

        print("Search evaluation results saved in path: %s" % save_path)


    @staticmethod
    def write_to_csv(results, save_path):
        with open(save_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for key, value in results.items():
                writer.writerow([key, value])

    @staticmethod
    def write_list_to_csv(items, save_path):
        with open(save_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for item in items:
                writer.writerow(item)


    @staticmethod
    def flatten(mapping):
        return np.array([item for items in mapping.values() for item in items])



