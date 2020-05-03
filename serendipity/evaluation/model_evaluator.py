import csv
import datetime

import numpy as np

from evaluation.evaluation import Evaluation


class ModelEvaluator(object):

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
        save_path = self.file_path + self.dataset + '/' + self.cell_name + str(idx) + '.evaluation_results.csv'

        self.write_to_csv(results, save_path)

        self.results_list.append(results)

        print("Evaluation results saved in path: %s" % save_path)

        evolution_results = [['key,precision,recall,mrr,diversity,aggregate_diversity,unexpectedness,novelty,herfindahl_diversity,entropy_diversity,precision_at_one,recall_at_one,mrr_at_one']]

        for key in sorted(precisions.keys()):
            precision_avg = np.average(precisions[key])
            recall_avg = np.average(recalls[key])
            mrr_avg = np.average(mrrs[key])
            diversity_avg = np.average(diversities[key])
            aggregate_diversity_avg = len(set(aggregate_diversities[key]))
            unexpectedness_avg = np.average(unexpectednesses[key])
            novelty_avg = np.average(novelties[key])
            herfindahl_avg = np.average(herfindahl_diversities[key])
            entropy_avg = np.average(entropy_diversities[key])
            precision_at_one_avg = np.average(precisions_at_one[key])
            recall_at_one_avg = np.average(recalls_at_one[key])
            mrr_at_one_avg = np.average(mrrs_at_one[key])

            evolution_result = [key, precision_avg, recall_avg, mrr_avg, diversity_avg,
                                aggregate_diversity_avg, unexpectedness_avg, novelty_avg, herfindahl_avg, entropy_avg, precision_at_one_avg, recall_at_one_avg, mrr_at_one_avg]

            evolution_results.append(evolution_result)

        evolution_save_path = self.file_path + self.dataset + '/' + self.cell_name + str(idx) + '.evolution_evaluation_results.csv'

        self.write_list_to_csv(evolution_results, evolution_save_path)

        self.evolution_results_list.append(evolution_results)

        print("Evolution evaluation results saved in path: %s" % evolution_save_path)

    def _get_keys(self):
        keys = set()
        for evolution_results in self.evolution_results_list:
            for evolution_result in evolution_results:
                if not isinstance(evolution_result[0], str):
                    keys.add(evolution_result[0])
        return sorted(keys)

    def _get_averages(self, key):
        precisions = []
        recalls = []
        mrrs = []
        diversities = []
        aggregate_diversities = []
        unexpectedness = []
        novelties = []
        herfindahl_diversities = []
        entropy_diversities = []
        precisions_at_one = []
        recalls_at_one = []
        mrrs_at_one = []

        for evolution_results in self.evolution_results_list:
            for evolution_result in evolution_results:
                if evolution_result[0] == key:
                    precisions.append(evolution_result[1])
                    recalls.append(evolution_result[2])
                    mrrs.append(evolution_result[3])
                    diversities.append(evolution_result[4])
                    aggregate_diversities.append(evolution_result[5])
                    unexpectedness.append(evolution_result[6])
                    novelties.append(evolution_result[7])
                    herfindahl_diversities.append(evolution_result[8])
                    entropy_diversities.append(evolution_result[9])
                    precisions_at_one.append(evolution_result[10])
                    recalls_at_one.append(evolution_result[11])
                    mrrs_at_one.append(evolution_result[12])
        return [
            key,
            np.average(precisions),
            np.average(recalls),
            np.average(mrrs),
            np.average(diversities),
            np.average(aggregate_diversities),
            np.average(unexpectedness),
            np.average(novelties),
            np.average(herfindahl_diversities),
            np.average(entropy_diversities),
            np.average(precisions_at_one),
            np.average(recalls_at_one),
            np.average(mrrs_at_one),
        ]

    def write_evolution_evaluation_results_average(self):
        save_path = self.file_path + self.dataset + '/' + self.cell_name + '.evolution_evaluation_results.csv'
        with open(save_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            head = 'key,precision,recall,mrr,diversity,aggregate_diversity,unexpectedness,novelty,herfindahl_diversity,entropy_diversity,precision_at_one,recall_at_one,mrr_at_one'
            writer.writerow(head.split(','))
            keys = self._get_keys()

            for key in keys:
                averages = self._get_averages(key)
                writer.writerow(averages)

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

    def write_results_average(self):
        self.write_evaluation_results_average()
        self.write_evolution_evaluation_results_average()




    def write_evaluation_results_average(self):
        save_path = self.file_path + self.dataset + '/' + self.cell_name + '.evaluation_results.csv'
        with open(save_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            first_item = self.results_list[0]
            for key in first_item.keys():
                average = np.average([result[key] for result in self.results_list])
                writer.writerow([key, average])

    @staticmethod
    def flatten(mapping):
        return np.array([item for items in mapping.values() for item in items])



