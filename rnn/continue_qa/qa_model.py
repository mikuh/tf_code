import tensorflow as tf
from tensorflow.contrib import rnn
import datetime, os
import data_help

class Attention_LSTM(object):
    """
    """
    def __init__(self, batch_size, lstm_size=500, pre_steps=100, query_steps=100, answer_steps=100, embedding_size=256,
                 dropout_keep_prob=0.8, evaluate_every=2000, checkpoint_every=2000, num_checkpoints=5,
                 max_steps=1000000, l2_reg_lambda=0.5, lr=0.01, max_grad_norm=10, train_data_file=None,
                 valid_data_file=None, checkpoint_dir='runs/checkpoints', is_train=True):
        self.batch_size = batch_size
        self.lstm_size = lstm_size
        self.pre_steps = pre_steps
        self.query_steps = query_steps
        self.answer_steps = answer_steps
        self.embedding_size = embedding_size
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.num_checkpoints = num_checkpoints
        self.max_steps = max_steps
        self.l2_reg_lambda = l2_reg_lambda
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.dropout_keep_prob = dropout_keep_prob
        self.session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        self.train_data_file = train_data_file
        self.valid_data_file = valid_data_file
        self.checkpoint_dir = checkpoint_dir
        self.is_train = is_train

    def inference(self, next_element):
        """forward propagation
        """
        pre_vec, query_vec, answer_vec = next_element[0], next_element[1], next_element[2]

        if self.is_train:
            neg_answer_vec = tf.concat([tf.slice(answer_vec, [1, 0, 0], [-1, -1, -1]), [answer_vec[0]]], 0)

        # set tensor shape
        pre_vec.set_shape([self.batch_size, self.pre_steps, self.embedding_size])
        query_vec.set_shape([self.batch_size, self.query_steps, self.embedding_size])

        if self.is_train:
            answer_vec.set_shape([self.batch_size, self.answer_steps, self.embedding_size])
            neg_answer_vec.set_shape([self.batch_size, self.answer_steps, self.embedding_size])

        # create attention and bi-lstm
        with tf.variable_scope("LSTM_scope"):
            pre = bi_lstm(pre_vec, self.lstm_size, self.pre_steps)

            pre_feat = tf.nn.tanh(max_pooling(pre))

        with tf.variable_scope("LSTM_scope", reuse=True):
            q = bi_lstm(query_vec, self.lstm_size, self.query_steps)
            question_feat = tf.nn.tanh(max_pooling(q))

        inputs = tf.concat([pre_feat, question_feat], 1)

        with tf.variable_scope("fc_layer"):
            inputs_enc = tf.contrib.layers.fully_connected(inputs, self.lstm_size * 2)

        U = tf.Variable(tf.truncated_normal([2 * self.lstm_size, self.embedding_size], stddev=0.1), name="U")

        with tf.variable_scope("LSTM_scope", reuse=True):
            pos_answer_att_weight = tf.sigmoid(tf.matmul(answer_vec,
                                                         tf.reshape(tf.expand_dims(tf.matmul(inputs_enc, U), 1),
                                                                    [-1, self.embedding_size, 1])))
            pos_a = bi_lstm(
                tf.multiply(answer_vec, tf.tile(pos_answer_att_weight, [1, 1, self.embedding_size])),
                self.lstm_size, self.answer_steps)
            pos_a_feat = tf.nn.tanh(max_pooling(pos_a))

            if self.is_train:
                neg_answer_att_weight = tf.sigmoid(tf.matmul(neg_answer_vec,
                                                             tf.reshape(tf.expand_dims(tf.matmul(question_feat, U), 1),
                                                                        [-1, self.embedding_size, 1])))

                neg_a = bi_lstm(
                    tf.multiply(neg_answer_vec, tf.tile(neg_answer_att_weight, [1, 1, self.embedding_size])),
                    self.lstm_size, self.answer_steps)
                neg_a_feat = tf.nn.tanh(max_pooling(neg_a))


        # dropout
        with tf.name_scope("dropout"):
            drop_q = tf.nn.dropout(inputs_enc, self.dropout_keep_prob)
            drop_a_pos = tf.nn.dropout(pos_a_feat, self.dropout_keep_prob)
            if self.is_train:
                drop_a_neg = tf.nn.dropout(neg_a_feat, self.dropout_keep_prob)

        question_vec = tf.identity(drop_q, name='q_vec')
        answer_vec = tf.identity(drop_a_pos, name='a_vec')
        pos_sim = tf.identity(feature2cos_sim(drop_q, drop_a_pos), name='cos_sim')
        if self.is_train:
            neg_sim = feature2cos_sim(drop_q, drop_a_neg)
        else:
            neg_sim = None
        return question_vec, answer_vec, pos_sim, neg_sim

    def loss(self, pos_sim, neg_sim):
        """loss function
        """
        zero = tf.constant(0, dtype=tf.float32)
        margin = tf.constant(0.5, dtype=tf.float32)
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(pos_sim, neg_sim)))
        loss = tf.reduce_sum(losses)
        return loss, losses

    def accurancy(self, losses):
        """accuracy of calculation
        """
        zero = tf.constant(0, dtype=tf.float32)
        correct_predictions = tf.equal(zero, losses)
        return tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    def train_step(self, sess, _handle):
        """Single step training
        """
        _, step, cur_loss, cur_acc = sess.run(
            [self.train_op, self.global_step, self._loss, self.acc], feed_dict={self.handle: _handle})
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_acc))

    def dev_step(self, sess, handle):
        """Signle step verifying
        """
        step, loss, accuracy = sess.run(
            [self.global_step, self._loss, self.acc], {self.handle: handle})
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}------dev".format(time_str, step, loss, accuracy))

    def train_operation(self):
        """train operation
        """
        # define training global steps
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), self.max_grad_norm)
        # define optimizer
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        return train_op

    def train(self):
        with tf.Graph().as_default():
            sess = tf.Session(config=self.session_conf)
            with sess.as_default():

                # load train and valid data
                train_iterator = data_help.get_iterator()
                valid_iterator = data_help.get_test_iterator()

                sess.run(train_iterator.initializer)
                sess.run(valid_iterator.initializer)

                train_iterator_handle = sess.run(train_iterator.string_handle())
                valid_iterator_handle = sess.run(valid_iterator.string_handle())

                self.handle = tf.placeholder(tf.string, shape=[], name='input_handle')

                iterator = tf.data.Iterator.from_string_handle(self.handle, train_iterator.output_types)

                next_element = iterator.get_next()

                _, _, pos_sim, neg_sim = self.inference(next_element)

                self._loss, _losses = self.loss(pos_sim, neg_sim)
                self.acc = self.accurancy(_losses)

                self.train_op = self.train_operation()

                # config for model saver
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
                print("Writing to {}\n".format(out_dir))

                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

                # Initialize all variables.
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())



                for _ in range(self.max_steps):
                    self.train_step(sess, train_iterator_handle)
                    cur_step = tf.train.global_step(sess, self.global_step)
                    # verify model and save model checkpoint
                    if cur_step % self.evaluate_every == 0 and cur_step != 0:
                        self.dev_step(sess, valid_iterator_handle)
                    if cur_step % self.checkpoint_every == 0 and cur_step != 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                        print("Saved model checkpoint to {}\n".format(path))

    def encoder(self, pre_place_holder, query_place_holder):
        with tf.variable_scope("LSTM_scope"):
            pre = bi_lstm(pre_place_holder, self.lstm_size, self.pre_steps)

            pre_feat = tf.nn.tanh(max_pooling(pre))

        with tf.variable_scope("LSTM_scope", reuse=True):
            q = bi_lstm(query_place_holder, self.lstm_size, self.query_steps)
            question_feat = tf.nn.tanh(max_pooling(q))

        inputs = tf.concat([pre_feat, question_feat], 1)

        with tf.variable_scope("fc_layer"):
            inputs_enc = tf.contrib.layers.fully_connected(inputs, self.lstm_size * 2)

        with tf.name_scope("dropout"):
            drop_q = tf.nn.dropout(inputs_enc, 1.0)

        return drop_q


def feature2cos_sim(feat_q, feat_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return cos_sim_q_a


def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)

    # do max-pooling to change the (sequence_length) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, width])
    return output


def avg_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)

    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.avg_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, width])
    return output


def bi_lstm(x, n_hidden, n_steps):
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, axis=1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                 dtype=tf.float32)
    # output transformation to the original tensor type
    outputs = tf.stack(outputs, axis=1)
    return outputs