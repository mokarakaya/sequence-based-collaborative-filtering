import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding

class MLPEmbedding(object):

    def __init__(self):

        self._num_items = None
        self.item_ids = None
        self.writer = tf.summary.FileWriter('/tmp/tensorflow')

    def _initialize(self, interactions):
        (self._num_users,
         self._num_items) = (interactions.num_users,
                             interactions.num_items)

        self.xs = tf.placeholder("float")
        self.ys = tf.placeholder("float")
        embedding_length = 32
        self.output = self._neural_net_model(self.xs, embedding_length * 2)

        self.cost = tf.reduce_mean(tf.pow(tf.transpose(self.output) - self.ys, 2))
        # tf_loss = tf.reduce_mean(tf.pow(tf_train_label - tf_prediction, 2))
        # beta = 0.001
        # self.cost = tf.reduce_mean(self.cost + beta * tf.nn.l2_loss(self.A2))

        self.train = tf.train.AdamOptimizer(0.001).minimize(self.cost)
        self.batch = 5000
        self.batch_size = 128
        self.sess = tf.Session()

        self.user_embedding = Sequential()
        self.user_embedding.add(Embedding(self._num_users, embedding_length, input_length=1))
        self.user_embedding.compile('rmsprop', 'mse')

        self.item_embedding = Sequential()
        self.item_embedding.add(Embedding(self._num_items, embedding_length, input_length=1))
        self.item_embedding.compile('rmsprop', 'mse')

        self.item_embedded = self.item_embedding.predict(np.arange(self._num_items))[:, 0]
        self.test = None
    def fit(self, interactions):
        self._initialize(interactions)

        # train = tf.train.GradientDescentOptimizer(0.005).minimize(cost)

        train_X = [[[0.1,0.2,0.3]], [[0.4,0.1,0.8]], [[0.2,0.1,0.7]], [[0.1,0.2,0.8]]]
        # train_X = [[[0.1,0.2,0.6]], [[0.4,0.1,0.9]], [[0.2,0.1,0.8]], [[0.1,0.2,20.0]]]
        train_y = [10, 20, 18, 19]
        user_ids = interactions.user_ids.astype(np.int64)
        item_ids = interactions.item_ids.astype(np.int64)
        ratings = interactions.ratings.astype(np.int64)

        one_hot_users = self.user_embedding.predict(user_ids)
        one_hot_items = self.item_embedding.predict(item_ids)

        train_X = tf.concat([one_hot_users[:,0], one_hot_items[:,0]], 1)

        self.writer.add_graph(self.sess.graph)
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver()
        train_X = self.sess.run(train_X)
        train_X_len = len(train_X)
        for i in range(self.batch):
            batch_indices = np.random.randint(0, train_X_len, size=self.batch_size)
            self.sess.run(self.train, feed_dict={
                self.xs: train_X[batch_indices], self.ys: ratings[batch_indices]})
            epoch_cost = self.sess.run(self.cost, feed_dict={
                self.xs: train_X[batch_indices], self.ys: ratings[batch_indices]})
            epoch_cost = epoch_cost
            print('Epoch :', i, 'Cost :', epoch_cost)

            # out = self.sess.run(self.output, feed_dict={
            #     self.xs: train_X[batch_indices], self.ys: ratings[batch_indices]})
            # a = np.power(out - ratings[batch_indices], 2)
            # print(np.average(a))
    def predict(self, user_ids, item_ids=None):

        # user_embedded = self.user_embedding.predict(np.full(self._num_items, user_ids))
        user_embeeded = self.user_embedding.predict([user_ids])[:, 0]
        user_embeeded = np.tile(user_embeeded, (self._num_items, 1))
        # user_embedded = tf.tile(self.user_embedding.predict([user_ids]), [self._num_items, 1, 1])
        # input_one_hot = tf.concat([user_embeeded, self.item_embedded], 1)
        stacked = np.column_stack((user_embeeded, self.item_embedded))

        predictions = self.sess.run(self.output, feed_dict={self.xs: stacked})
        return predictions[:, 0]

    def _neural_net_model(self, X_data, input_dim):


        hidden_layer_nodes = 100
        A1 = tf.Variable(
            tf.truncated_normal(shape=[input_dim, hidden_layer_nodes]))  # inputs -> hidden nodes
        b1 = tf.Variable(
            tf.zeros(shape=[hidden_layer_nodes]))  # one biases for each hidden node
        self.A2 = tf.Variable(
            tf.truncated_normal(shape=[hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
        b2 = tf.Variable(tf.zeros(shape=[1]))  # 1 bias for the output

        hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, A1), b1))
        final_output = tf.add(tf.matmul(hidden_output, self.A2), b2)

        return final_output

