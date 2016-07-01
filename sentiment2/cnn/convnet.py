
import tensorflow as tf
import numpy as np

class ConvNet(object):

    def __init__(self, seq_lenth, num_classes, vocab_size, embedding_dim, filter_lengths, num_filters):

        self.input_x = tf.placeholder(tf.int32, [None, seq_lenth], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep = tf.placeholder(tf.float32, name="dropout_keep")

        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='W')
            self.embedded_word = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_word_expanded = tf.expand_dims(self.embedded_word, -1)

        pool_out = []
        for length in filter_lengths:
            with tf.name_scope("conv-maxpool-" + str(length)):
                filter_shape = [length, embedding_dim, 1, num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')

                conv = tf.nn.conv2d(self.embedded_word_expanded, W, strides=[1,1,1,1], padding='VALID', name='conv')

                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                
                max_pool = tf.nn.max_pool(
                        h,
                        ksize=[1, seq_lenth - length + 1, 1, 1],
                        strides=[1,1,1,1],
                        padding='VALID',
                        name='maxpool')
                pool_out.append(max_pool)

        flat_len = num_filters * len(filter_lengths)
        self.h_pool = tf.concat(3, pool_out)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, flat_len])

        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep)

        with tf.name_scope('output'):
            W = tf.get_variable(
                    'W',
                    shape=[flat_len, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.pred = tf.argmax(self.scores, 1, name='pred')

        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"), name="accuracy")
