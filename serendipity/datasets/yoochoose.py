"""
Utilities for fetching the Movielens datasets [1]_.

References
----------

.. [1] https://grouplens.org/datasets/movielens/
"""

import os

import h5py

from spotlight.datasets import _transport
from spotlight.interactions import Interactions

VARIANTS = ('clicks',
            'buys')


URL_PREFIX = ('https://github.com/maciejkula/recommender_datasets/'
              'releases/download')
VERSION = 'v0.2.0'


def _get_yoochoose(dataset):

    extension = '.hdf5'

    path = _transport.get_data('/'.join((URL_PREFIX,
                                         VERSION,
                                         dataset + extension)),
                               os.path.join('yoochoose', VERSION),
                               'yoochoose_{}{}'.format(dataset,
                                                       extension))

    with h5py.File(path, 'r') as data:
        return (data['/user_id'][:],
                data['/item_id'][:],
                None,
                data['/timestamp'][:])


def get_yoochoose_dataset(variant='buys'):

    if variant not in VARIANTS:
        raise ValueError('Variant must be one of {}, '
                         'got {}.'.format(VARIANTS, variant))

    dataset = 'yoochoose_{}'.format(variant)

    return Interactions(*_get_yoochoose(dataset))
