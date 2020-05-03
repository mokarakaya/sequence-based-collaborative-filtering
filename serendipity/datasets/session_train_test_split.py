import numpy as np
from spotlight.interactions import Interactions


def _get_interactions(interactions, indices):
    return Interactions(interactions.user_ids[indices],
                        interactions.item_ids[indices],
                        timestamps=interactions.timestamps[indices],
                        num_users=interactions.num_users,
                        num_items=interactions.num_items)


def _create_interactions(user_ids, item_ids, timestamps,
                         num_users, num_items):
    return Interactions(user_ids,
                        item_ids,
                        timestamps=timestamps,
                        num_users=num_users,
                        num_items=num_items)


def _remove_new_items(test, train):
    item_ids = test.item_ids
    user_ids = test.user_ids
    timestamps = test.timestamps

    new_items = set(test.item_ids) - set(train.item_ids)

    for new_item in new_items:
        indices = np.where(item_ids == new_item)
        item_ids = np.delete(item_ids, indices)
        user_ids = np.delete(user_ids, indices)
        timestamps = np.delete(timestamps, indices)

    return Interactions(user_ids,
                        item_ids,
                        timestamps=timestamps,
                        num_users=train.num_users,
                        num_items=train.num_items)


def _train_test_split(interactions_list, test_interval):
    train_test_list = []
    for interactions in interactions_list:
        timestamps = interactions.timestamps
        max_timestamp = timestamps.max()
        indices = np.digitize(timestamps, np.array([0, max_timestamp - test_interval, max_timestamp + 1]))
        train_indices = np.argwhere(indices == 1)[:, 0]
        test_indices = np.argwhere(indices == 2)[:, 0]
        train = _get_interactions(interactions, train_indices)
        test = _get_interactions(interactions, test_indices)

        test = _remove_new_items(test, train)

        seq_user_ids, seq_item_ids, user_mapping, item_mapping = _convert_to_sequential(train.user_ids, train.item_ids)
        seq_user_ids_test, seq_item_ids_test, _, _ = _convert_to_sequential(test.user_ids, test.item_ids,
                                                                            user_mapping, item_mapping)

        print('seq_item_ids:', max(seq_item_ids), ',', 'seq_item_ids_test:', max(seq_item_ids_test))

        assert max(seq_item_ids) >= max(seq_item_ids_test)


        num_users = max(max(seq_user_ids), max(seq_user_ids_test)) + 1
        num_items = max(seq_item_ids) + 1
        train_seq = _create_interactions(seq_user_ids,
                                         seq_item_ids,
                                         train.timestamps,
                                         num_users,
                                         num_items)
        test_seq = _create_interactions(seq_user_ids_test,
                                        seq_item_ids_test,
                                        test.timestamps,
                                        num_users,
                                        num_items)
        train_test_list.append((train_seq, test_seq))
    return train_test_list


def _get_sequential_id(idx, count, mapping):
    if idx not in mapping:
        count += 1
        mapping[idx] = count
    return mapping[idx], count


def _convert_to_sequential(user_ids, item_ids,
                           user_mapping=None, item_mapping=None):
    if user_mapping is None:
        user_mapping = {}
        count_user_id = 0
    else:
        count_user_id = max(user_mapping.values())

    if item_mapping is None:
        item_mapping = {}
        count_item_id = 0
    else:
        count_item_id = max(item_mapping.values())


    seq_user_ids = []
    seq_item_ids = []
    for user_id, item_id in zip(user_ids, item_ids):
        seq_user_id, count_user_id = _get_sequential_id(user_id, count_user_id, user_mapping)
        seq_item_id, count_item_id = _get_sequential_id(item_id, count_item_id, item_mapping)
        seq_user_ids.append(seq_user_id)
        seq_item_ids.append(seq_item_id)
    return np.asarray(seq_user_ids), np.asarray(seq_item_ids), user_mapping, item_mapping


def session_based_train_test_split(interactions,
                                   number_of_windows=5,
                                   test_interval_days=1):
    num_users = interactions.num_users
    num_items = interactions.num_items
    test_interval = test_interval_days * 24 * 60 * 60
    timestamps = interactions.timestamps
    max_timestamp = timestamps.max()
    min_timestamp = timestamps.min()
    timestamps_histogram = np.histogram(timestamps,
                                              bins=number_of_windows,
                                              range=(min_timestamp, max_timestamp + 1))
    timestamps_indices = np.digitize(timestamps, timestamps_histogram[1])
    interactions_list = []
    for i in range(1, number_of_windows + 1):
        args = np.argwhere(timestamps_indices == i)[:, 0]
        user_ids = interactions.user_ids[args]
        item_ids = interactions.item_ids[args]

        interactions_list.append(Interactions(user_ids,
                                              item_ids,
                                              timestamps=interactions.timestamps[args],
                                              num_users=num_users,
                                              num_items=num_items))
    return _train_test_split(interactions_list, test_interval)
