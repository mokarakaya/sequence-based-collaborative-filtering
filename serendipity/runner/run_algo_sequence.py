
import os
import sys


import datetime
import numpy as np
from spotlight.datasets import movielens
from spotlight.cross_validation import user_based_train_test_split
from serendipity.deep.rnn_embedding import RNNEmbedding
from serendipity.evaluation.model_evaluator import ModelEvaluator
from serendipity.datasets.yoochoose import get_yoochoose_dataset
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.datasets import goodbooks, amazon
from serendipity.datasets.session_train_test_split import session_based_train_test_split
import serendipity.datasets.dataset_sessionization as ds
from collections import Counter
from serendipity.deep.GRU4Rec.gru4rec_adaptor import GRU4RecAdaptor

def run():
    RANDOM_STATE = np.random.RandomState(42)
    CUDA = bool(os.environ.get('SPOTLIGHT_CUDA', True))
    print('start run', datetime.datetime.now())

    k = 20
    max_sequence_length = 50
    min_sequence_length = 2
    # max_sequence_length = 20
    # min_sequence_length = 5
    step_size = 1

    # max_sequence_length = 200
    # min_sequence_length = 50
    # step_size = 50

    # datasets = {
    #     'Movielens100K': lambda: movielens.get_movielens_dataset('100K'),
    #     'Movielens1M', 'yoochoose_buys', 'yoochoose_clicks']
    # }

    datasets_to_run = ['nowplaying', 'retailrocket', 'yoochoose_clicks']
    # datasets_to_run = ['nowplaying']

    datasets = {
        'Movielens100K': {'dataset': lambda: ds.get_sessionized_movielens_dataset('100K')},
        'Movielens1M': {'dataset': lambda: ds.get_sessionized_movielens_dataset('1M')},
        'yoochoose_buys': {'dataset': lambda: ds.get_yoochoose_dataset('buys')},
        'yoochoose_clicks': {'dataset': lambda: ds.get_yoochoose_dataset('clicks'), 'num_gates': 10, 'num_hidden': 100},
        'goodbooks': {'dataset': lambda: ds.get_sessionized_goodbooks_dataset()},
        'amazon': {'dataset': lambda: ds.get_sessionized_amazon_dataset()},
        'nowplaying': {'dataset': lambda: ds.get_sessionized_nowplaying_dataset(), 'num_gates': 3, 'num_hidden': 10},
        'retailrocket': {'dataset': lambda: ds.get_sessionized_retailrocket_dataset(), 'num_gates': 5, 'num_hidden': 50},
    }


    def get_most_common(session_ids):
        counter = Counter(session_ids)
        most_common = counter.most_common(1)
        return most_common[0][1]


    # cell_names = ['LSTMCellDiversity', 'BasicRNNCell','LSTMCell', 'GRUCell','GRU4Rec', 'DBAM']
    # cell_names = ['LSTMCell', 'GRUCell', 'GRU4Rec', 'DBAM']
    cell_names = ['DBAM']
    # cell_names = ['LSTMCellDiversity', 'BasicRNNCell','LSTMCell', 'DBAM', 'GRUCell']
    # cell_names = ['LSTMCellDiversity', 'BasicRNNCell','LSTMCell', 'GRUCell', 'DBAM']
    # cell_names = ['LSTMCellDiversity', 'BasicRNNCell','LSTMCell', 'DBAM']
    # cell_names = ['BasicRNNCell','LSTMCell', 'GRUCell', 'DBAM']
    for dataset in datasets_to_run:
        interactions = datasets[dataset]['dataset']()
        train_test_list = session_based_train_test_split(interactions,
                                                         number_of_windows=4,
                                                         test_interval_days=1)
        num_gates = datasets[dataset]['num_gates']
        num_hidden = datasets[dataset]['num_hidden']

        for cell_name in cell_names:
            # if cell_name == 'LSTMCellDiversity':
            #     num_hidden = 50

            model_evaluator = ModelEvaluator(cell_name, dataset)
            for idx, (train, test) in enumerate(train_test_list):

                if idx != 0 and dataset == 'yoochoose_clicks':
                    continue
                train_max_sequence_length = get_most_common(train.user_ids)
                test_max_sequence_length = get_most_common(test.user_ids)

                max_sequence_length = min(max_sequence_length, max(train_max_sequence_length, test_max_sequence_length))

                test_seq = test.to_sequence(max_sequence_length=max_sequence_length,
                                            min_sequence_length=2,
                                            step_size=step_size)

                if cell_name == 'GRU4Rec':
                    model = GRU4RecAdaptor(idx, dataset)
                    model.fit(train, verbose=False)
                else:
                    train_seq = train.to_sequence(
                        max_sequence_length=max_sequence_length,
                        min_sequence_length=min_sequence_length,
                        step_size=step_size)
                    # import pdb; pdb.set_trace()
                    print('train_seq.shape', train_seq.sequences.shape)
                    model = RNNEmbedding(idx, debug_enabled=False, cell_name=cell_name, dataset=dataset,
                                         num_gates=num_gates, num_hidden=num_hidden, save_restore_models=True)
                    model.fit(train_seq, verbose=False)
                model_evaluator.evaluate_model(model, train, test, test_seq, idx)
            model_evaluator.write_results_average()

run()
