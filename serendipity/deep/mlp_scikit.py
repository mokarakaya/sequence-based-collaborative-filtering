from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class MLPScikit(object):

    def __init__(self):

        self._num_items = None
        self.item_ids = None

    def _initialize(self, interactions):

        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)
        self.mlp_regressor = MLPRegressor(hidden_layer_sizes=(100,), verbose=True, max_iter=200)
        self.label_encoder_user = LabelEncoder()
        self.onehot_encoder_user = OneHotEncoder(sparse=False)

        self.label_encoder_item = LabelEncoder()
        self.onehot_encoder_item = OneHotEncoder(sparse=False)


        self.users_encoded = False
        self.items_encoded = False

        self.onehot_encoded_items = None

    def fit(self, interactions):
        self._initialize(interactions)
        self.onehot_encoded_items = self._one_hot_encode_item(np.arange(self._num_items))

        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)
        ratings = interactions.ratings.astype(np.int64)

        onehot_encoded_users = self._one_hot_encode_user(user_ids)
        onehot_encoded_items = self._one_hot_encode_item(item_ids)
        X = np.column_stack((onehot_encoded_users, onehot_encoded_items))

        batch_size = 10000

        for idx in range(1):
            epochs = int(len(X) / batch_size)
            for epoch in range(epochs):
                epoch_start = epoch * batch_size
                epoch_end = (epoch + 1) * batch_size
                epoch_x = X[epoch_start:epoch_end]
                epoch_y = ratings[epoch_start:epoch_end]

                self.mlp_regressor.partial_fit(epoch_x, epoch_y)

                predict_y = self.mlp_regressor.predict(X)
                error = np.abs(predict_y - ratings)
                print(np.sum(error) / len(ratings))

    def predict(self, user_ids, item_ids=None):

        _user_ids = np.full((self._num_items,), user_ids)
        onehot_encoded_users = self._one_hot_encode_user(_user_ids)
        X = np.column_stack((onehot_encoded_users, self.onehot_encoded_items))
        predict_y = self.mlp_regressor.predict(X)
        return predict_y

    def _one_hot_encode_user(self, user_ids):
        if not self.users_encoded:
            integer_encoded_users = self.label_encoder_user.fit_transform(user_ids)
            integer_encoded_users = integer_encoded_users.reshape(len(integer_encoded_users), 1)
            onehot_encoded_users = self.onehot_encoder_user.fit_transform(integer_encoded_users)

            self.users_encoded = True
            return onehot_encoded_users
        else:
            integer_encoded_users = self.label_encoder_user.transform(user_ids)
            integer_encoded_users = integer_encoded_users.reshape(len(integer_encoded_users), 1)
            onehot_encoded_users = self.onehot_encoder_user.transform(integer_encoded_users)

            return onehot_encoded_users

    def _one_hot_encode_item(self, item_ids):
        if not self.items_encoded:
            integer_encoded_items = self.label_encoder_item.fit_transform(item_ids)
            integer_encoded_items = integer_encoded_items.reshape(len(integer_encoded_items), 1)
            onehot_encoded_items = self.onehot_encoder_item.fit_transform(integer_encoded_items)

            self.items_encoded = True
            return onehot_encoded_items
        else:
            integer_encoded_items = self.label_encoder_item.transform(item_ids)
            integer_encoded_items = integer_encoded_items.reshape(len(integer_encoded_items), 1)
            onehot_encoded_items = self.onehot_encoder_item.transform(integer_encoded_items)
            return onehot_encoded_items
