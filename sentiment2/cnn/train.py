
import tensorflow as tf
import numpy as np

import datetime

from preprocess import DataSet, read_validation, ValidationSetReader

from convnet import ConvNet

tf.flags.DEFINE_integer("embedding_dim", 128, "")
tf.flags.DEFINE_string("filter_lengths", "3,4,5,6", "")
tf.flags.DEFINE_integer("num_filters", 128, "")
tf.flags.DEFINE_float("dropout_keep", 0.5, "")

tf.flags.DEFINE_integer("batch_size", 64, "")
tf.flags.DEFINE_integer("num_epochs", 200, "")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "")
tf.flags.DEFINE_boolean("log_device_placement", False, "")

FLAGS = tf.flags.FLAGS

FLAGS._parse_flags()

pos_samples = './twitter-datasets/train_pos.txt'
neg_samples = './twitter-datasets/train_neg.txt'
vocab_file = './vocab.dat'
inv_vocab_file = './inv_vocab.dat'
valid_target_x = './xval.txt'
valid_target_y = './yval.txt'

data = DataSet(
        pos_samples,
        neg_samples,
        vocab_file,
        inv_vocab_file,
        valid_target_x,
        valid_target_y)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = ConvNet(
                seq_lenth=data.seq_len,
                num_classes=2,
                vocab_size=len(data.vocab),
                embedding_dim=FLAGS.embedding_dim,
                filter_lengths=[int(l) for l in FLAGS.filter_lengths.split(',')],
                num_filters=FLAGS.num_filters)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            feed_dict = {
                    cnn.input_x : x_batch,
                    cnn.input_y : y_batch,
                    cnn.dropout_keep : FLAGS.dropout_keep
                    }

            _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(time_str + ' ' + str(step) + ' ' + str(accuracy))

        def eval_step(x_batch, y_batch):
            feed_dict = {
                    cnn.input_x : x_batch,
                    cnn.input_y : y_batch,
                    cnn.dropout_keep : 1.0
                    }
            step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)

            print("accuracy on validation set: " + str(accuracy))

        while True:
            try:
                x_batch, y_batch = next(data.batches(FLAGS.batch_size))
            except StopIteration:
                break

            train_step(x_batch, y_batch)

        validation_reader = ValidationSetReader(valid_target_x, valid_target_y)
        while True:
            try:
                x_val, y_val = read_validation(valid_target_x, valid_target_y)
            except StopIteration:
                break
            
            eval_step(x_val, y_val)

