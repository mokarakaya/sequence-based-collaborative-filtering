
import pandas as pd
import numpy as np

class PopularityRec:

    def __init__(self,
                 idx,
                 dataset,
                 file_path='../../../sequence-based-collaborative-filtering-models/'):

        self.recommend = None

    def fit(self, interactions, verbose=True):
        df = pd.DataFrame(interactions.item_ids, columns=['item_ids'])
        df = df.groupby('item_ids').item_ids.count()
        df = df.transform(lambda x: x/x.max())
        # df = pd.DataFrame(interactions.item_ids, columns=['item_ids'])
        # self.recommend = np.array(df.groupby('item_ids').item_ids.count().sort_values(ascending=False)[:20].keys())
        self.recommend = np.insert(np.array(df.values), 0, 0)

    def predict(self, sequences, user_id, item_ids=None):
        return self.recommend


class RandomRec:

    def __init__(self,
                 idx,
                 dataset,
                 file_path='../../../sequence-based-collaborative-filtering-models/'):

        self.num_items = None

    def fit(self, interactions, verbose=True):
        self.num_items = interactions.num_items

    def predict(self, sequences, user_id, item_ids=None):
        return np.random.rand(self.num_items)

