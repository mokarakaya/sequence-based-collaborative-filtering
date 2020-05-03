import datetime
import os

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import tensorflow.contrib.losses as loss_ops
import numpy as np
from tensorflow.python import debug as tf_debug

from deep.DiverstiyBasedAttentionMechanism.rnn_cell import DistractionLSTMCell_soft, DistractionLSTMCell_hard, DistractionLSTMCell_subtract
from serendipity.deep.lstm_diversity import LSTMCellDiversity
from serendipity.deep.dualLSTM import DualLSTMCell
from sklearn.metrics.pairwise import cosine_similarity
import sys
import math
from serendipity.io_ops.model_operator import ModelOperator

class RNNEmbedding(object):

    def __init__(self,
                 id,
                 debug_enabled=False,
                 cell_name='LSTMCellDiversity',
                 dataset=None,
                 val_size=0.2,
                 file_path='../../../sequence-based-collaborative-filtering-models/',
                 num_gates=2,
                 save_restore_models=True,
                 num_hidden=100):

        tf.reset_default_graph()
        self.id = id
        self.debug_enabled = debug_enabled
        self.cell_name = cell_name
        self.val_size = val_size
        self.learning_rate = 0.01
        self.training_steps = 50
        # self.training_steps = 0
        self.batch_size = 64
        self.display_step = 1
        self.keep_prob = 0.9
        self.lambda_l2_reg = 1e-10
        # self.lambda_l2_reg = 1e-5
        # self.lambda_l2_reg = 1e-2

        self.writer = tf.summary.FileWriter('/tmp/tensorflow', tf.get_default_graph())
        self.file_path = file_path + dataset + '/'+cell_name + str(id)
        self.loss_op_summary = None


        self.test_loss = sys.float_info.max
        self.model_operator = ModelOperator(self.file_path)
        self.num_gates = num_gates
        self.save_restore_models = save_restore_models
        self.num_hidden = num_hidden

    def _initialize(self, interactions):
        self._num_items = interactions.num_items
        self.max_sequence_length = interactions.max_sequence_length

        self.embedding_size = 32
        self.num_input = self.embedding_size  # MNIST data input (img shape: 28*28)
        self.timesteps = self.max_sequence_length  # timesteps
        self.num_hidden = self.num_hidden  # hidden layer num of features
        self.num_classes = self._num_items  # MNIST total classes (0-9 digits)

        # self.embedding = Sequential()
        # self.embedding.add(Embedding(self._num_items, self.embedding_size,
        #                              input_length=self.max_sequence_length - 1,
        #                              name="item_embeddings_scope"))
        # self.embedding.compile('rmsprop', 'mse')

        # TODO: can be removed in tensorflow 1.12 ?
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # self.run_metadata = tf.RunMetadata()
        self.sess = tf.Session(config=config)

        if self.debug_enabled:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type="readline")


    def fit(self, interactions, verbose=False, sparse = True):
        self._initialize(interactions)

        self.X = tf.placeholder(dtype=tf.int32, shape=[None, self.timesteps - 1], name='X_noreg')
        # item_ids = tf.placeholder("float32", [None, self.timesteps - 1, self.num_input], name='X_noreg')
        if not sparse:
            Y = tf.placeholder("float32", [None, self.num_classes], name='Y_noreg')
        else:
            Y = tf.placeholder("int32", [None], name='Y_noreg')

        print('self._num_items', self._num_items)
        mlp_P = tf.Variable(tf.random_normal([self._num_items, self.embedding_size]), dtype=tf.float32)
        embedding_layer = tf.nn.embedding_lookup(mlp_P, self.X)

        # Define weights
        weights = {
            'out': tf.Variable(tf.truncated_normal([self.num_hidden, self.num_classes]),
                               name='diversity_lstm_input_weights')
        }
        biases = {
            'out': tf.Variable(tf.truncated_normal([self.num_classes]),
                               name='diversity_lstm_input_biases')
        }

        self.logits = self._rnn_model(embedding_layer, weights, biases)
        self.prediction = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        if not sparse:
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=Y))
        # loss_op = tf.reduce_mean(loss_ops.hinge_loss(
        #     self.logits, labels=Y))
        else:
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=Y))

        l2 = self.lambda_l2_reg * sum(
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
        # self.saver = tf.train.Saver()
        self.writer.add_graph(self.sess.graph)

        if self.save_restore_models:
            self.sess = self.model_operator.read_model(self.sess)



        # import pdb; pdb.set_trace()
        sequences = interactions.sequences
        sequences_length = sequences.shape[0]
        val_count = sequences_length * self.val_size
        val_samp = np.random.randint(low=1, high=sequences_length - 1, size=(int(val_count),))

        mask = np.ones(sequences_length, dtype=bool)
        mask[val_samp] = False

        val = sequences[~mask]
        train = sequences[mask]

        val_x = val[:, :-1]
        val_y = val[:, - 1]
        early_stopping = 0

        for step in range(1, self.training_steps + 1):
            print(datetime.datetime.now())

            # print('validation loss:', valid_loss)

            np.random.shuffle(train)
            train_X = train[:, :-1]
            train_Y = train[:, - 1]

            if not sparse:
                train_Y = tf.one_hot(train_Y, self._num_items)
                train_Y = self.sess.run(train_Y)
            num_of_batches = len(train_X) / self.batch_size

            self.train_batches(Y, num_of_batches, step, train_X, train_Y, train_op)

            valid_loss = self.get_validation_loss(Y, loss_op, val_x, val_y)
            model_improved = self.model_operator.save_model(self.sess, valid_loss, self.save_restore_models)
            # model_improved = True
            print('validation loss:', valid_loss)
            if model_improved:
                early_stopping = 0
            else:
                early_stopping += 1

            if (math.isnan(valid_loss) or valid_loss == 0.0) and step >= 5:
            # if step >= 5:
                print('nan and enough steps:', valid_loss, step)
                return
            if early_stopping == 2:
                print('model does not improve for 2 iterations')
                return


        print("Optimization Finished!")
        # save_path = self.saver.save(self.sess, self.file_path)
        # print("Model saved in path: %s" % save_path)

    def get_validation_loss(self, Y, loss_op, val_x, val_y):
        total_loss = 0.0
        counter = 0.0
        print('validation:', len(val_x), self.batch_size)
        for i in range(0, len(val_x), self.batch_size):
            batch_x = val_x[i:i + self.batch_size]
            batch_y = val_y[i:i + self.batch_size]
            loss = self.sess.run(
                loss_op,
                feed_dict={self.X: batch_x, Y: batch_y})
            counter += 1
            # print(loss, ':', i)
            if not math.isnan(loss):
                total_loss += loss
        if total_loss == 0.0:
            print('could not calculate loss :', total_loss, counter)
        if counter < 1.0:
            print('counter did not iterate:', counter)
        return total_loss / counter

    def train_batches(self, Y, num_of_batches, step, train_X, train_Y, train_op):
        for i in range(0, len(train_X), self.batch_size):
            # batch_indices = np.random.randint(0, train_X_len, size=self.batch_size)

            batch_x = train_X[i:i + self.batch_size]
            batch_y = train_Y[i:i + self.batch_size]
            # Reshape data to get 28 seq of 28 elements
            # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            self.sess.run(train_op,
                          feed_dict={self.X: batch_x, Y: batch_y})
            # options=self.run_options,
            # run_metadata=self.run_metadata)
            # print("step:", step, 'batch:', i, 'num_of_batches', num_of_batches)

    def predict(self, sequences, user_id, item_ids=None):
        sequences_filled = np.hstack([np.zeros(self.max_sequence_length - 1 - len(sequences)), sequences])
        # embedded_seq = self.embedding.predict(np.array([sequences_filled]))

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
        elif self.cell_name == 'DBAM':
            cell = DistractionLSTMCell_soft(self.num_hidden, state_is_tuple=True)
        elif self.cell_name == 'DualLSTMCell':
            cell = DualLSTMCell(self.num_hidden, forget_bias=1.0, name='lstm_cell_diversity')
        else:
            cell = LSTMCellDiversity(self.num_hidden, forget_bias=1.0, name='lstm_cell_diversity', num_gates=self.num_gates)
            # cell = DualLSTMCell(self.num_hidden, forget_bias=1.0, name='lstm_cell_diversity')

        cell = rnn.DropoutWrapper(cell, state_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        # Get lstm cell output
        self.outputs, self.states = rnn.static_rnn(cell, x, dtype=tf.float32)
        # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(self.outputs[-1], weights['out']) + biases['out']

