import datetime
import os

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import tensorflow.contrib.losses as loss_ops
import numpy as np
from tensorflow.python import debug as tf_debug
from sklearn.preprocessing import OneHotEncoder
# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

class RNNOneHot(object):

    def __init__(self,
                 debug_enabled=False,
                 cell_name='BasicRNNCell',
                 test_seq=None):

        self.debug_enabled = debug_enabled
        self.cell_name = cell_name
        self.learning_rate = 1e-3
        self.training_steps = 1
        self.batch_size = 32
        self.display_step = 10

        self.writer = tf.summary.FileWriter('/tmp/tensorflow', tf.get_default_graph())
        self.file_path = '/tmp/rnn_one_hot/'+cell_name+'.ckpt'
        self.loss_op_summary = None
        self.test_seq = test_seq

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self.max_sequence_length = interactions.max_sequence_length

        self.embedding_size = self._num_items
        self.num_input = self.embedding_size  # MNIST data input (img shape: 28*28)
        self.timesteps = self.max_sequence_length  # timesteps
        self.num_hidden = 10  # hidden layer num of features
        self.num_classes = self._num_items  # MNIST total classes (0-9 digits)

        # TODO: can be removed in tensorflow 1.12 ?
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # self.run_metadata = tf.RunMetadata()
        self.sess = tf.Session(config=config)

        if self.debug_enabled:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type="readline")

        encoder = OneHotEncoder()
        items = np.arange(self._num_items).reshape((self._num_items, 1))

        self.one_hot_items = encoder.fit_transform(items).toarray()

    def fit(self, interactions):
        self._initialize(interactions)


        self.X = tf.placeholder("float", [None, self.timesteps - 1, self.num_input], name='X_noreg')
        Y = tf.placeholder("float", [None, self.num_classes], name='Y_noreg')

        # Define weights
        weights = {
            'out': tf.Variable(tf.truncated_normal([self.num_hidden, self.num_classes]),
                               name='diversity_lstm_input_weights')
        }
        biases = {
            'out': tf.Variable(tf.truncated_normal([self.num_classes]),
                               name='diversity_lstm_input_biases')
        }

        self.logits = self._rnn_model(self.X, weights, biases)
        self.prediction = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=Y))
        # loss_op = tf.reduce_mean(loss_ops.hinge_loss(
        #     self.logits, labels=Y))

        lambda_l2_reg = 0.009
        l2 = lambda_l2_reg * sum(
            tf.nn.l2_loss(tf_var)
            for tf_var in tf.trainable_variables()
            if not ("noreg" in tf_var.name or "bias" in tf_var.name or tf_var.dtype == np.int64)
        )
        loss_op += l2

        self.loss_op_summary = tf.summary.scalar('loss_op_'+self.cell_name, loss_op)

        self.test_loss_op_summary = tf.summary.scalar('test loss_op_'+self.cell_name, loss_op)

        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # train_X_len = len(train_X)

        # Run the initializer
        self.sess.run(init)
        self.saver = tf.train.Saver()
        self.writer.add_graph(self.sess.graph)
        if os.path.exists(self.file_path + '.index'):
            self.saver.restore(self.sess, self.file_path)
            print("Model restored.")
        else:
            print("Model restore skipped.")

        test_sequences = self.test_seq.sequences
        test_X = test_sequences[:, :-1]
        test_X = self.one_hot_items[test_X]
        test_X = self.sess.run(test_X)

        test_Y = test_sequences[:, - 1]
        test_Y = tf.one_hot(test_Y, self._num_items)
        test_Y = self.sess.run(test_Y)

        for step in range(1, self.training_steps + 1):
            sequences = interactions.sequences
            np.random.shuffle(sequences)
            train_X = sequences[:, :-1]
            # import pdb;
            # pdb.set_trace()

            # train_X = tf.one_hot(train_X, self._num_items)
            # train_X = self.sess.run(train_X)

            train_Y = sequences[:, - 1]
            train_Y = tf.one_hot(train_Y, self._num_items)
            train_Y = self.sess.run(train_Y)

            for i in range(0, len(train_X), self.batch_size):

                # batch_indices = np.random.randint(0, train_X_len, size=self.batch_size)

                batch_x = train_X[i:i + self.batch_size]
                batch_x = tf.one_hot(batch_x, self._num_items)
                batch_x = self.sess.run(batch_x)

                batch_y = train_Y[i:i + self.batch_size]
                # Reshape data to get 28 seq of 28 elements
                # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                # Run optimization op (backprop)
                self.sess.run(train_op,
                              feed_dict={self.X: batch_x, Y: batch_y})
                              # options=self.run_options,
                              # run_metadata=self.run_metadata)

                if i == 0 and (step % self.display_step == 0 or step == 1):
                    loss, acc, loss_summary = self.sess.run(
                        [loss_op, accuracy, self.loss_op_summary],
                        feed_dict={self.X: batch_x, Y: batch_y})

                    test_loss, test_accuracy, test_loss_summary = self.sess.run(
                        [loss_op, accuracy, self.test_loss_op_summary],
                        feed_dict={self.X: test_X, Y: test_Y})

                    print("Step " + str(step))
                    print(", Minibatch Loss= " + "{:.4f}".format(loss) +
                          ", Training Accuracy= " + "{:.3f}".format(acc))
                    print(", Test Minibatch Loss= " + "{:.3f}".format(test_loss) +
                          ", Test Accuracy= " + "{:.3f}".format(test_accuracy))
                    print(datetime.datetime.now())
                    self.writer.add_summary(loss_summary, step)
                    self.writer.add_summary(test_loss_summary, step)

        print("Optimization Finished!")
        save_path = self.saver.save(self.sess, self.file_path)
        print("Model saved in path: %s" % save_path)

    def predict(self, sequences, item_ids=None):
        sequences_filled = np.hstack([np.zeros(self.max_sequence_length - 1 - len(sequences)), sequences])
        sequences_filled = tf.one_hot(sequences_filled, self._num_items)
        sequences_filled = self.sess.run(sequences_filled)

        predictions = self.sess.run(self.prediction , feed_dict={self.X: [sequences_filled]})
        return predictions[0]

    def _rnn_model(self, x, weights, biases):

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(x, self.timesteps - 1, axis=1)


        if self.cell_name == 'BasicRNNCell':
            cell = rnn.BasicRNNCell(self.num_hidden)
        elif self.cell_name == 'LSTMCell':
            cell = tf.nn.rnn_cell.LSTMCell(self.num_hidden, forget_bias=1.0, name='basic_lstm_cell')
        elif self.cell_name == 'GRUCell':
            cell = tf.nn.rnn_cell.GRUCell(self.num_hidden, name='gru_cell')


        cell = rnn.DropoutWrapper(cell, state_keep_prob=0.8, output_keep_prob=0.8)
        # Get lstm cell output
        outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)
        # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

