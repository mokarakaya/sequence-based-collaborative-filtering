import datetime
import time
import copy


def go_to_previous_time(train):

    back_to = 3
    max_timestamp = max(train.timestamps)
    max_date = datetime.date.fromtimestamp(max_timestamp)
    max_date - datetime.timedelta(days=back_to * 30)
    mktime = int(time.mktime(max_date.timetuple()))
    indexes = [n for n, x in enumerate(train.timestamps) if x < mktime]
    train_prev = copy.deepcopy(train)
    train_prev.user_ids = train_prev.user_ids[indexes]
    train_prev.item_ids = train_prev.item_ids[indexes]
    train_prev.ratings = train_prev.ratings[indexes]
    train_prev.timestamps = train_prev.timestamps[indexes]
    return train_prev
