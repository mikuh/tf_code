"""
预先训练词向量的ptb模型
"""
import tensorflow as tf
import data_helper
import os
import datetime
import numpy as np


class PTBModel(object):
    """
    ptb模型
    """

    def __init__(self, hparams, initial_lr, lr_decay_factor, num_examples, max_grad_norm=15, emb=False,
                 is_training=True, num_epochs_per_decay=5, num_checkpoints=5):
        self.batch_size = batch_size = hparams.batch_size           # 批次大小
        self.num_steps = num_steps = hparams.num_steps              # lstm时间步数
        self.unit_size = hparams.unit_size                          # lstm单元神经元个数
        self.num_layers = hparams.num_layers                        # lstm层数
        self.vocab_size = hparams.vocab_size                        # 词向量维度
        self.initial_lr = initial_lr                                # 最初的学习速率
        self.lr_decay_factor = lr_decay_factor                      # 学习速率衰减因子
        self.max_grad_norm = max_grad_norm                          # 限制最大梯度
        self.emb = emb                                              # 是否词嵌入
        self.is_training = is_training                              # 是否在训练
        self.num_checkpoints = num_checkpoints                      # 保存点个数
        self.epoch_size = ((len(num_examples) // batch_size) - 1) // num_steps
        self.num_epochs_per_decay = num_epochs_per_decay

    def build_rnn_graph(self, inputs, keep_prob):
        """构建lstm"""
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            self.unit_size, forget_bias=0.0, state_is_tuple=True,
            reuse=not self.is_training)

        if self.is_training and keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)

        # 多层lstm
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(self.num_layers)], state_is_tuple=True)
        # 初始状态为0
        initial_state = state = stacked_lstm.zero_state(self.batch_size, tf.float32)

        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                # 后面的时间步数复用第一步的权重变量
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = stacked_lstm(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, self.unit_size])  # 将时间步数连接起来
        final_state = state
        return output, final_state

    def inference(self, inputs, keep_prob):
        """向前传播
        """
        # 词嵌入
        if self.emb:
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [self.vocab_size, self.unit_size], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(embedding, inputs)

        # 丢弃神经元
        if self.is_training and keep_prob < 1.0:
            inputs = tf.nn.dropout(inputs, keep_prob)

        output, final_state = self.build_rnn_graph(inputs, keep_prob)

        if self.emb:
            softmax_w = tf.get_variable("softmax_w", [self.unit_size, self.vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size], dtype=tf.float32)
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        else:
            logits = tf.reshape(output, [self.batch_size, self.num_steps, self.vocab_size])

        return logits


    def loss(self, logits, targets):
        """损失函数
        """
        loss = tf.contrib.seq2seq.sequence_loss(
            logits, targets,
            tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
        return tf.reduce_sum(loss, name='loss')

    def train_operation(self, loss, global_step):
        """训练操作
        """
        decay_steps = int(self.epoch_size * self.num_epochs_per_decay)
        lr = tf.train.exponential_decay(self.initial_lr, global_step, decay_steps, self.lr_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('loss', loss)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        return train_op

    def train_step(self, sess, summary_writer,):
        """单批次训练
        """
        _, step, cur_loss, cur_acc = sess.run([self.train_op, self.global_step, self._loss, self.accuracy])
        self.costs += cur_loss
        self.iters += self.num_steps
        train_perplexity = np.exp(self.costs / self.iters)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, perplexity {:g} acc {:g}".format(time_str, step, cur_loss, train_perplexity, cur_acc))
        # 存储摘要
        if step % 100 == 0:
            summary_str = sess.run(self.summary)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()


    def train(self, filename, out_dir):
        """训练
        """
        with tf.Graph().as_default():
            sess = tf.Session(config=self.session_conf)
            with sess.as_default():
                self.global_step = tf.contrib.framework.get_or_create_global_step()
                with tf.device('/cpu:0'):
                    inputs, targets = data_helper.distorted_inputs(filename, self.batch_size)
                logits = self.inference(inputs, )
                self._loss = self.loss(logits, targets)
                self.train_op = self.train_operation(self._loss, self.global_step)
                self.accuracy = self.evaluation(logits, targets)
                self.summary = tf.summary.merge_all()

                # 保存点设置
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")  # 模型存储前缀
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)
                summary_writer = tf.summary.FileWriter(out_dir + "/summary", sess.graph)

                # 初始化所有变量
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())

                tf.train.start_queue_runners(sess=sess)
                self.costs = 0
                self.iters = 0
                for step in range(self.max_steps):
                    self.train_step(sess, summary_writer)  # 训练
                    cur_step = tf.train.global_step(sess, self.global_step)
                    # checkpoint_every 次迭代之后 保存模型
                    if cur_step % self.checkpoint_every == 0 and cur_step != 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                        print("Saved model checkpoint to {}\n".format(path))