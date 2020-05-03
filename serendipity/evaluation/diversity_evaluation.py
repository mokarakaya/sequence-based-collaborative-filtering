import numpy as np
import scipy.sparse as sp
from serendipity.neighborhood.similarity import similarity_matrix
import sklearn.preprocessing as pp
FLOAT_MAX = np.finfo(np.float32).max


class DiversityEvaluation(object):

    def __init__(self, train, train_prev=None):
        self._sim = similarity_matrix(train.tocsr().T)
        self._sim_prev = None
        if train_prev:
            similarity_matrix(train_prev.tocsr().T)

    @staticmethod
    def usefulness_score(model, test, train, k=20):
        """
        average ratings of recommended items.
        """
        distances = []
        test = test.tocsr()
        train = train.tocsr()
        (x, y, z) = sp.find(train.T)
        counts = np.bincount(x)
        counts += 1
        sums = np.bincount(x, weights=z)
        averages = sums / counts

        for user_id, row in enumerate(test):

            if not len(row.indices):
                continue

            predictions = -model.predict(user_id)

            if train is not None:
                rated = train[user_id].indices
                predictions[rated] = FLOAT_MAX

            rec_list = predictions.argsort()[:k]
            distances.append(np.mean(averages[rec_list]))
        return np.array(distances)

    @staticmethod
    def _distance(_sim, first_item, second_item):
        return 1 - _sim[first_item][second_item]

    def usefulness_time_score(self, model, test, train, train_prev, k=20):
        distances = []
        test = test.tocsr()
        mat = train.tocsr()
        mat_prev = train_prev.tocsr()
        for user_id, row in enumerate(test):
            predictions = -model.predict(user_id)
            rec_list = predictions.argsort()[:k]
            user_interactions = mat[user_id].indices
            user_interactions_prev = mat_prev[user_id].indices
            distance = [
                self._distance(self._sim, first_item, second_item)
                for i, first_item in enumerate(rec_list)
                for second_item in user_interactions
            ]
            distance_prev = [
                self._distance(self._sim_prev, first_item, second_item)
                for i, first_item in enumerate(rec_list)

                for second_item in user_interactions_prev
            ]
            distance_avg = np.nan_to_num(np.mean(distance))
            distance_avg_prev = np.nan_to_num(np.mean(distance_prev))
            distances.append((1 + distance_avg_prev - distance_avg) / 2)
        return np.array(distances)

    def usefulness_weighted_sim_score(self, model, test, train, k=20):
        usefulness = []
        test = test.tocsr()
        mat = train.tocsr()

        popularity = np.bincount(mat.indices)
        popularity = pp.minmax_scale(popularity)
        item_ids = mat.indices
        ratings = mat.data
        counts = np.bincount(item_ids)
        counts += 1
        sums = np.bincount(item_ids, weights=ratings)
        averages = sums / counts
        # averages = pp.minmax_scale(averages)

        sims = []
        pops = []
        avgs = []
        for user_id, row in enumerate(test):

            if not len(row.indices):
                continue

            predictions = -model.predict(user_id)

            if mat is not None:
                rated = mat[user_id].indices
                predictions[rated] = FLOAT_MAX

            rec_list = predictions.argsort()[:k]
            user_interactions = mat[user_id].indices
            sim, pop, avg = zip(*[
                self._usefulness_weighted(first_item, second_item, self._sim, popularity, averages)
                for i, first_item in enumerate(rec_list)
                for second_item in user_interactions
            ])

            weighted_sim = np.array(sim) - (np.array(pop) * 0.5)
            # usefulness.append(np.mean(weighted_sim * weighted_sim * avg))
            usefulness.append(np.mean(np.abs(weighted_sim) * avg))
            sims.append(np.mean(np.array(sim)))
            pops.append(np.mean(np.array(pop)))
            avgs.append(np.mean(np.array(avg)))

        # return np.array(usefulness)
        return np.array(usefulness), sims, pops, avgs
    @staticmethod
    def _usefulness_weighted(first_item, second_item, sim, popularity, averages):
        return sim[first_item][second_item], popularity[first_item], averages[first_item]

    def serendipity_time_score(self, model, test, train, train_prev, k=20):
        distances = []
        test = test.tocsr()
        mat = train.tocsr()
        mat_prev = train_prev.tocsr()

        for user_id, row in enumerate(test):
            predictions = -model.predict(user_id)
            rec_list = predictions.argsort()[:k]
            user_interactions = mat[user_id].indices
            user_interactions_prev = mat_prev[user_id].indices
            distance = [
                self._distance(self._sim, first_item, second_item)
                for i, first_item in enumerate(rec_list)
                for second_item in user_interactions
            ]
            distance_prev = [
                self._distance(self._sim_prev, first_item, second_item)
                for i, first_item in enumerate(rec_list)
                for second_item in user_interactions_prev
            ]
            distance_avg = np.nan_to_num(np.mean(distance))
            distance_avg_prev = np.nan_to_num(np.mean(distance_prev))
            distances.append(distance_avg * (1 + distance_avg_prev - distance_avg) / 2)
        return np.array(distances)

    def serendipity_score(self, model, test, train, k=20):
        distances = []
        test = test.tocsr()
        mat = train.tocsr()
        (x, y, z) = sp.find(mat.T)
        counts = np.bincount(x)
        counts += 1
        sums = np.bincount(x, weights=z)
        averages = sums / counts
        for user_id, row in enumerate(test):

            if not len(row.indices):
                continue

            predictions = -model.predict(user_id)

            if mat is not None:
                rated = mat[user_id].indices
                predictions[rated] = FLOAT_MAX

            rec_list = predictions.argsort()[:k]
            user_interactions = mat[user_id].indices
            distance = [
                (1 - self._sim[first_item][second_item]) * averages[first_item]
                for i, first_item in enumerate(rec_list)
                for second_item in user_interactions
            ]
            distances.append(np.nan_to_num(np.mean(distance)))
        return np.array(distances)

    def unexpectedness_score(self, model, test, train, k=20):
        distances = []
        test = test.tocsr()
        mat = train.tocsr()
        for user_id, row in enumerate(test):

            if not len(row.indices):
                continue

            predictions = -model.predict(user_id)

            if mat is not None:
                rated = mat[user_id].indices
                predictions[rated] = FLOAT_MAX

            rec_list = predictions.argsort()[:k]
            user_interactions = mat[user_id].indices
            distance = [
                1 - self._sim[first_item][second_item]
                for i, first_item in enumerate(rec_list)
                for second_item in user_interactions
            ]
            distances.append(np.nan_to_num(np.mean(distance)))
        return np.array(distances)

    def intra_distance_score(self, model, test, k=20):
        distances = []
        test = test.tocsr()
        for user_id, row in enumerate(test):
            if not len(row.indices):
                continue

            predictions = -model.predict(user_id)
            rec_list = predictions.argsort()[:k]
            distance = [
                1 - self._sim[first_item][second_item]
                for i, first_item in enumerate(rec_list)
                for second_item in rec_list[(i + 1):]
            ]
            distances.append(distance)
        return np.array(distances)
