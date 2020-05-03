import pandas as pd
# from sklearn.cross_validation import train_test_split
import numpy as np
import numpy as np

import tensorflow as tf

import sys
from serendipity.io_ops.model_operator import ModelOperator


class MLPOneHot(object):

    def __init__(self,
                 id,
                 debug_enabled=False,
                 cell_name='MLPOneHot',
                 dataset=None,
                 test_seq=None,
                 file_path='../../../sequence-based-collaborative-filtering-models/'):

        tf.reset_default_graph()
        self.id = id
        self.debug_enabled = debug_enabled
        self.cell_name = cell_name
        self.learning_rate = 1e-3
        self.training_steps = 10
        self.batch_size = 64
        self.display_step = 1
        self.keep_prob = 0.8
        self.lambda_l2_reg = 1e-10
        self.num_hidden = 20  # hidden layer num of features
        # self.lambda_l2_reg = 1e-5
        # self.lambda_l2_reg = 1e-2

        self.writer = tf.summary.FileWriter('/tmp/tensorflow', tf.get_default_graph())
        self.file_path = file_path + dataset + '/'+cell_name + str(id)
        self.loss_op_summary = None
        self.test_seq = test_seq

        self.test_loss = sys.float_info.max
        self.model_operator = ModelOperator(self.file_path)

    def _initialize(self, interactions):

        self.n_input = interactions.sequences.max() + 1
        n_classes = self.n_input
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.num_hidden])),
            'out': tf.Variable(tf.random_normal([self.num_hidden, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self.num_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        logits = self.multilayer_perceptron(self.X, weights, biases)
        self.prediction = tf.nn.softmax(logits)
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(loss_op)
        # Initializing the variables
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # self.run_metadata = tf.RunMetadata()
        self.sess = tf.Session(config=config)

    # Create model
    def multilayer_perceptron(self, x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    def fit(self, interactions):
        self._initialize(interactions)
        # Run the initializer
        init = tf.global_variables_initializer()

        # train_X_len = len(train_X)

        # Run the initializer
        self.sess.run(init)
        # self.saver = tf.train.Saver()
        self.writer.add_graph(self.sess.graph)

        self.sess = self.model_operator.read_model(self.sess)

        test_sequences = self.test_seq.sequences
        test_X = test_sequences[:, :-1]
        # test_X = self.embedding.predict(test_X)

        test_Y = test_sequences[:, - 1]

        for step in range(1, self.training_steps + 1):
            sequences = interactions.sequences
            np.random.shuffle(sequences)
            train_X = sequences[:, :-1]
            train_X = self.create_one_hot(train_X)
            # train_X = self.embedding.predict(train_X)

            train_Y = sequences[:, - 1]
            train_Y = self.create_one_hot(train_Y)

            for i in range(0, len(train_X), self.batch_size):

                # batch_indices = np.random.randint(0, train_X_len, size=self.batch_size)

                batch_x = train_X[i:i + self.batch_size]
                batch_y = train_Y[i:i + self.batch_size]
                # Reshape data to get 28 seq of 28 elements
                # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                # Run optimization op (backprop)
                self.sess.run(self.train_op,
                              feed_dict={self.X: batch_x, self.Y: batch_y})
                # options=self.run_options,
                # run_metadata=self.run_metadata)
                # import pdb;
                # pdb.set_trace()



        print("Optimization Finished!")
        # save_path = self.saver.save(self.sess, self.file_path)
        # print("Model saved in path: %s" % save_path)


    def create_one_hot(self, sequences):
        one_hot_sequences = []
        for sequence in sequences:
            one_hot = np.zeros(self.n_input)
            one_hot[sequence] = 1
            one_hot[0] = 0
            one_hot_sequences.append(one_hot)
        return one_hot_sequences

    def predict(self, sequences, user_id, item_ids=None):
        # if self.predictions is None:
        #     self.one_hot_users = tf.one_hot([user_ids], self._num_users + 1)[0]
        # tf.one_hot([user_ids], self._num_users + 1)
        one_hot = np.zeros(self.n_input )
        one_hot[sequences] = 1
        one_hot[0] = 0

        predictions = self.sess.run(self.prediction, feed_dict={self.X: [one_hot]})
        return predictions[0]

        # one_hot_users = tf.one_hot([user_ids], self._num_users + 1)[0]
        # user_embeeded = np.tile(self.sess.run(one_hot_users), (self._num_items, 1))
        # stacked = np.column_stack((user_embeeded, self.one_hot_items))
        # predictions = self.sess.run(self.output, feed_dict={self.xs: stacked})
        # return predictions[:, 0]

    def _neural_net_model(self, X_data, input_dim):

        hidden_layer_nodes = 100
        self.A1 = tf.Variable(
            tf.truncated_normal(shape=[input_dim, hidden_layer_nodes]))  # inputs -> hidden nodes
        b1 = tf.Variable(
            tf.zeros(shape=[hidden_layer_nodes]))  # one biases for each hidden node
        self.A2 = tf.Variable(
            tf.truncated_normal(shape=[hidden_layer_nodes, 1]))  # hidden inputs -> 1 output
        b2 = tf.Variable(tf.zeros(shape=[1]))  # 1 bias for the output

        hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, self.A1), b1))
        final_output = tf.add(tf.matmul(hidden_output, self.A2), b2)

        return final_output

