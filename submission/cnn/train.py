
import tensorflow as tf
import numpy as np

import datetime

from preprocess import DataSet, read_validation, ValidationSetReader, PredictionSet

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

pos_samples = './data/train_pos_full.txt'
neg_samples = './data/train_neg_full.txt'
#pos_samples = './data/pos_small.txt'
#neg_samples = './data/neg_small.txt'
vocab_file = './data/vocab.dat'
inv_vocab_file = './data/inv_vocab.dat'
valid_target_x = './data/xval.txt'
valid_target_y = './data/yval.txt'
pred_file = './data/pred.txt'
test_data = './data/test_data.txt'

data = DataSet(
        pos_samples,
        neg_samples,
        vocab_file,
        inv_vocab_file,
        valid_target_x,
        valid_target_y,
        hold_out_proba=0.0)

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

            _, step, loss, accuracy, pred = sess.run([train_op, global_step, cnn.loss, cnn.accuracy, cnn.pred], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print(time_str + ' ' + str(step) + ' ' + str(accuracy))

        def eval_step(x_batch, y_batch):
            feed_dict = {
                    cnn.input_x : x_batch,
                    cnn.input_y : y_batch,
                    cnn.dropout_keep : 1.0
                    }
            step, loss, accuracy, pred = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.pred], feed_dict)

            print("accuracy on validation set: " + str(accuracy))

        def pred_step(x_batch, y_batch):
            feed_dict = {
                    cnn.input_x : x_batch,
                    cnn.input_y : y_batch,
                    cnn.dropout_keep : 1.0
                    }
            step, loss, accuracy, pred = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.pred], feed_dict)

            with open(pred_file, 'a') as f:
                for p in pred:
                    if p == 1:
                        o = 1
                    elif p == 0:
                        o = -1
                    else:
                        raise Exception('wtf?')

                    f.write(str(o) + '\n')

        while True:
            try:
                x_batch, y_batch = next(data.batches(FLAGS.batch_size))
            except StopIteration:
                break

            train_step(x_batch, y_batch)

        #validation_reader = ValidationSetReader(valid_target_x, valid_target_y)
        #while True:
        #    try:
        #        x_val, y_val = next(validation_reader.batches(FLAGS.batch_size))
        #    except StopIteration:
        #        break
        #    
        #    eval_step(x_val, y_val)

        pred_set = PredictionSet(test_data, data.vocab, data.seq_len, DataSet.PAD_WORD)
        while True:
            try:
                x_batch = next(pred_set.batches(FLAGS.batch_size))
            except StopIteration:
                break

            # dummy y values
            y_batch = np.vstack([np.array([0,1]) for i in range(0, len(x_batch))])

            pred_step(x_batch, y_batch)
        

