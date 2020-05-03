import pandas as pd
from spotlight.interactions import Interactions
import numpy as np
import os
import csv
from spotlight.datasets import movielens
from serendipity.datasets.yoochoose import get_yoochoose_dataset
from serendipity.datasets.data_util import get_nowplaying_dataset
from serendipity.datasets.data_util import get_retailrocket_dataset
from spotlight.datasets import goodbooks, amazon

DATA_DIR = os.path.join(os.path.expanduser('~'),
                        'sequence-based-collaborative-filtering')
extension = '.csv'


def _get_sessionized_dataset(data_dir, dataset_fn, idle_time_mins):
    data_dir += str(idle_time_mins) + extension
    if os.path.isfile(data_dir):
        df = pd.read_csv(data_dir)
        return Interactions(df.values[:, 0], df.values[:, 1], timestamps=df.values[:, 2])
    else:
        interactions = dataset_fn()
        interactions = _sessionize_dataset(interactions, idle_time_mins)
        with open(data_dir, 'w') as writeFile:
            writer = csv.writer(writeFile)
            data = [interactions.user_ids, interactions.item_ids, interactions.timestamps]
            writer.writerows(zip(*data))
        return interactions


def get_sessionized_movielens_dataset(variant='100K', idle_time_mins=30):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), 'Movielens' + variant)
    return _get_sessionized_dataset(data_dir, lambda: movielens.get_movielens_dataset(variant), idle_time_mins)


def get_sessionized_goodbooks_dataset(idle_time_mins=30):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), 'goodbooks')
    return _get_sessionized_dataset(data_dir, lambda: goodbooks.get_goodbooks_dataset(), idle_time_mins)


def get_sessionized_amazon_dataset(idle_time_mins=30):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), 'amazon')
    return _get_sessionized_dataset(data_dir, lambda: amazon.get_amazon_dataset(), idle_time_mins)


def get_sessionized_yoochoose_dataset(variant='buys', idle_time_mins=30):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), 'yoochoose' + variant)
    return _get_sessionized_dataset(data_dir, lambda: get_yoochoose_dataset(variant), idle_time_mins)

def get_sessionized_nowplaying_dataset(idle_time_mins=30):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), 'nowplaying')
    return _get_sessionized_dataset(data_dir, lambda: get_nowplaying_dataset(), idle_time_mins)

def get_sessionized_retailrocket_dataset(idle_time_mins=30):
    data_dir = os.path.join(os.path.abspath(DATA_DIR), 'retailrocket')
    return _get_sessionized_dataset(data_dir, lambda: get_retailrocket_dataset(), idle_time_mins)


def _sessionize_dataset(interactions, idle_time_mins):

    idle_time = idle_time_mins * 60 * 1000
    df = pd.DataFrame({'user_ids': interactions.user_ids,
                       'item_ids': interactions.item_ids,
                       'timestamps': interactions.timestamps})
    df = df.sort_values(['user_ids', 'timestamps'])
    session_id = 0
    prev_user_id = 0
    prev_timestamp = 0
    session_ids = []
    records = df.to_records()
    for record in records:
        _, user_id, item_id, timestamp = record
        if user_id != prev_user_id or abs(timestamp - prev_timestamp) > idle_time:
            session_id += 1
        prev_user_id = user_id
        prev_timestamp = timestamp
        session_ids.append(session_id)
    return Interactions(np.array(session_ids), interactions.item_ids, timestamps=interactions.timestamps)
