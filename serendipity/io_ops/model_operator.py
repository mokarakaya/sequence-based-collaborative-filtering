import sys
import csv
import pickle
import os
import tensorflow as tf
import math


class ModelOperator(object):

    def __init__(self, file_path):
        self.test_loss = sys.float_info.max
        self.file_path = file_path
        self.saver = None

    def read_model(self, sess):
        self._init_saver()
        if os.path.exists(self.file_path + '.ckpt.index'):
            self.saver.restore(sess, self.file_path + '.ckpt')
            print("Model restored.")
            self.test_loss = self._read_test_loss()
            print('Test loss restored.')
        else:
            print("Model restore skipped.")
        return sess

    def save_model(self, sess, test_loss_new, save_restore_models):
        self._init_saver()
        if self.test_loss >= test_loss_new or math.isnan(test_loss_new):
            print("Better model found")
            # return
            self.test_loss = test_loss_new
            if save_restore_models:
                save_path = self.saver.save(sess, self.file_path + '.ckpt')
                with open(self.file_path + '.test_loss.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow({self.test_loss})
                print("Model saved in path: %s" % save_path)
            return True
        return False

    def _init_saver(self):
        if self.saver is None:
            self.saver = tf.train.Saver()

    def _read_test_loss(self):
        path = self.file_path + '.test_loss.csv'
        reader = csv.reader(open(path))
        for line in reader:
            return float(line[0])
        assert EOFError('test loss not found in file' + path)

    def save_object(self, model):
        file_handler = open(self.file_path, 'wb')
        pickle.dump(model, file_handler)

    def read_object(self):
        if os.path.exists(self.file_path):
            file_handler = open(self.file_path, 'rb')
            return pickle.load(file_handler)
        return None
