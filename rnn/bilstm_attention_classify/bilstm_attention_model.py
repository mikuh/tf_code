#coding=utf-8
from __future__ import print_function, unicode_literals
import tensorflow as tf
from tensorflow.contrib import rnn
import datetime
import os
class BilstmAttention(object):
    """双向LSTM+Attention
    """
    def __init__(self, num_steps, embedding_size, num_classes, lstm_size, max_grad_norm, learning_rate, attention_size,
                 num_checkpoints, evaluate_every, checkpoint_every, dropout_keep_prob,
                 batch_size=None, l2_reg_lambda=0.0, is_training=True):
        self.num_steps = num_steps
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.lstm_size = lstm_size
        self.l2_loss = tf.constant(0.0)
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.attention_size = attention_size
        self.num_checkpoints = num_checkpoints
        self.evaluate_every = evaluate_every
        self.checkpoint_every = checkpoint_every
        self.dropout_keep_prob_value = dropout_keep_prob
        self.batch_size = batch_size
        self.l2_reg_lambda = l2_reg_lambda
        self.is_training = is_training


    def placeholder_inputs(self):
        """创建占位符"""
        self.inputs_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_steps, self.embedding_size), name='inputs')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_classes), name='targets')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def droupout(self, inputs):
        """丢弃神经元
        """
        dropout_inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
        return dropout_inputs

    def bi_lstm(self, x, num_units):
        """双向LSTM
        """
        x = tf.unstack(x, axis=1)
        lstm_fw_cell = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
        lstm_bw_cell = rnn.BasicLSTMCell(num_units, forget_bias=1.0)
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        outputs = tf.stack(outputs, axis=1)
        return outputs


    def attention(self, inputs):
        """注意力机制
        """
        # 双向lstm返回正向和反向两个结果 存在一个tuplue中，要先整合到一起
        # 就是将两个向量 拼接起来
        if isinstance(inputs, tuple):
            inputs = tf.concat(2, inputs)
        sequence_length = inputs.get_shape()[1].value
        hidden_size = inputs.get_shape()[2].value

        # 注意力计算
        W_omega = tf.get_variable("W_omega", initializer=tf.random_normal([hidden_size, self.attention_size], stddev=0.1))
        b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([1, self.attention_size], stddev=0.1))
        u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([self.attention_size, 1], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega) + b_omega)
        vu = tf.matmul(v, u_omega)
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        return output

    def inference(self, inputs):
        """前向传播
        """
        # droupout
        if self.is_training:
            dropout_outputs = self.droupout(inputs)
        # bi-lstm
        with tf.name_scope('bi_lstm'):
            bi_lstm_outputs = self.bi_lstm(dropout_outputs, self.lstm_size)
        # attention
        with tf.name_scope('attention'):
            attention_outputs = self.attention(bi_lstm_outputs)
        # fully connect
        with tf.name_scope("output"):
            softmax_w = tf.get_variable("softmax_w", initializer=tf.truncated_normal([2*self.lstm_size, self.num_classes], stddev=0.1))
            softmax_b = tf.get_variable("softmax_b", initializer=tf.constant(0.05, shape=[self.num_classes]))
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)
            self.logits = tf.nn.xw_plus_b(attention_outputs, softmax_w, softmax_b, name="logits")        # 预测值
            self.class_value = tf.nn.softmax(self.logits, 1, name='class_value')         # 分类概率
            self.predictions = tf.argmax(self.logits, 1, name="predictions")             # 分类下标
        return self.logits

    def loss(self, logits, labels):
        """损失函数
        """
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        self.loss = tf.identity(tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss, name='loss')

    def train_ope(self):
        """训练操作
        :param loss: 损失值
        :param learning_rate: 学习速率
        :return: train_op: 训练做操句柄
        """
        tf.summary.scalar('loss', self.loss)  # 生成汇总值的操作
        self.global_step = tf.Variable(0, name="global_step", trainable=False)  # 步数
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)  # 限制梯度
        # 定义优化器
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)  # 训练操作

    def evaluation(self, logits, labels, k=1):
        """评估函数
        :param logits: 预测
        :param labels: 标签
        """
        correct = tf.nn.in_top_k(logits, tf.argmax(labels, 1), k=k)
        # correct = tf.equal(self.predictions, tf.argmax(labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    def train_step(self, sess, x, y, summary_writer):
        """单步训练
        """
        feed_dict = {
            self.inputs_placeholder: x,
            self.labels_placeholder: y,
            self.dropout_keep_prob: self.dropout_keep_prob_value
        }
        _, step, cur_loss, cur_acc = sess.run([self.train_op, self.global_step, self.loss, self.accuracy], feed_dict=feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_acc))
        # 存储摘要
        if step % 100 == 0:
            summary_str = sess.run(self.summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

    def dev_step(self, sess, x, y):
        """验证模型
        """
        feed_dict = {
            self.inputs_placeholder: x,
            self.labels_placeholder: y,
            self.dropout_keep_prob: 1.0
        }
        step, cur_loss, cur_acc = sess.run([self.global_step, self.loss, self.accuracy], feed_dict)
        print("验证模型：")
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_acc))


    def train(self, sess, batches, valid_batches, out_dir):
        """训练模型
        """
        self.placeholder_inputs()
        self.inference(self.inputs_placeholder)
        self.loss(self.logits, self.labels_placeholder)
        self.train_ope()
        self.evaluation(self.logits, self.labels_placeholder)
        self.summary = tf.summary.merge_all()

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
        summary_writer = tf.summary.FileWriter(out_dir+"/summary", sess.graph)

        # 初始化所有变量
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        for train_tuple in batches:
            self.train_step(sess, train_tuple[0], train_tuple[1], summary_writer)
            cur_step = tf.train.global_step(sess, self.global_step)
            # evaluate_every 次迭代之后 验证 并且 保存模型
            if cur_step % self.evaluate_every == 0 and cur_step != 0:
                valid_tuple = next(valid_batches)
                self.dev_step(sess, valid_tuple[0], valid_tuple[1])
            if cur_step % self.checkpoint_every == 0 and cur_step != 0:
                path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                print("Saved model checkpoint to {}\n".format(path))






