from serendipity.deep.GRU4Rec.gru4rec import GRU4Rec
import pandas as pd
import numpy as np
from serendipity.io_ops.model_operator import ModelOperator
from random import randrange

class GRU4RecAdaptor:

    def __init__(self,
                 idx,
                 dataset,
                 file_path='../../../sequence-based-collaborative-filtering-models/'):

        self.file_path = file_path + dataset + '/GRU4Rec' + str(idx)
        self.model_operator = ModelOperator(self.file_path)
        self.gru4rec = None
        self.dataset = dataset

    def fit(self, interactions, verbose=True):
        self.gru4rec = self.model_operator.read_object()
        if self.gru4rec is None:
            self._build_model(interactions)
        else:
            print('gru4rec: n_epochs:', self.gru4rec.n_epochs, ' loss:', self.gru4rec.loss)

    def _build_model(self, interactions):
        # self.gru4rec = GRU4Rec(loss='bpr-max',
        #                        final_act='elu-0.5',
        #                        hidden_act='tanh',
        #                        layers=[100],
        #                        adapt='adagrad',
        #                        n_epochs=200,
        #                        batch_size=32,
        #                        dropout_p_embed=0,
        #                        dropout_p_hidden=0,
        #                        learning_rate=0.2,
        #                        momentum=0.3,
        #                        n_sample=2048,
        #                        sample_alpha=0,
        #                        bpreg=1,
        #                        constrained_embedding=False)

        if self.dataset == 'nowplaying':
            self.gru4rec = GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[25], adapt='adagrad',
                                   n_epochs=20, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0.1,
                                   learning_rate=0.1,
                                   momentum=0.5, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
        elif self.dataset == 'retailrocket':
            self.gru4rec = GRU4Rec(loss='top1-max', final_act='elu-0.5', hidden_act='tanh', layers=[50],
                                   adapt='adagrad',
                                   n_epochs=30, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0,
                                   learning_rate=0.15,
                                   momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
        else:
            self.gru4rec = GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                                   n_epochs=30, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                                   momentum=0.5, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)


        df = pd.DataFrame({'SessionId': interactions.user_ids,
                           'ItemId': interactions.item_ids,
                           'Time': interactions.timestamps})
        self.gru4rec.fit(df)
        self.model_operator.save_object(self.gru4rec)

    def predict(self, sequences, user_id, item_ids=None):
        # self.gru4rec.predict = None
        items = np.trim_zeros(sequences)
        predictions = None
        for item in items:
            predictions = self.gru4rec.predict_next_batch([user_id], [item], item_ids, batch=1)

        predictions = predictions.values[:, 0]
        return np.insert(predictions, 0, 0)
