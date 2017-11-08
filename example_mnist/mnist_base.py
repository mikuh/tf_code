import tensorflow as tf



class Mnist(object):
    def __init__(self, flags):
        """基本网络结构参数
        """
        self.num_class = flags.num_class
        self.image_pixels = flags.image_size * flags.image_size
        self.batch_size = flags.batch_size
        self.hidden1_units = flags.hidden1_units
        self.hidden2_units = flags.hidden2_units
        self.max_step = self.batch_size * flags.num_epochs


    def placeholder_inputs(self, batch_size=None):
        """创建占位符
        """
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, self.image_pixels), name='input')
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        return images_placeholder, labels_placeholder

    def inference(self, images, hidden1_units, hidden2_units):
        """前向传播
        :param hidden1_units:  第一个隐藏层的神经元个数
        :param hidden2_units:  第二个隐藏层的神经元个数
        :return: 回归结果
        """
        # Hidden 1
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([self.image_pixels, hidden1_units],
                                    stddev=1.0 / tf.sqrt(float(self.image_pixels))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')
            hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                    stddev=1.0 / tf.sqrt(float(hidden1_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]),
                                 name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, self.num_class],
                                    stddev=1.0 / tf.sqrt(float(hidden2_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.num_class]),
                                 name='biases')
            logits = tf.matmul(hidden2, weights) + biases
        return logits


    def loss(self, logits, labels):
        """损失函数
        :param logits: 预测
        :param labels: 标签
        :return: 交叉熵损失
        """
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')


    def train(self, loss, learning_rate):
        """训练操作
        :param loss: 损失值
        :param learning_rate: 学习速率
        :return: train_op: 训练做操句柄
        """
        tf.summary.scalar('loss', loss)  # 生成汇总值得操作
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op

    def evaluation(self, logits, labels):
        """评估函数
        :param logits: 预测
        :param labels: 标签
        :return: top-k 正确数量
        """
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label is in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32), name='accuracy')

    def fill_feed_dict(self, data_set, images_pl, labels_pl, fake_data):
        """给占位符喂数据
        :param data_set:  数据集
        :param images_pl: 样本
        :param labels_pl: 标签
        :return:  喂数据的字典
        """
        # Create the feed_dict for the placeholders filled with the next
        # `batch size` examples.
        images_feed, labels_feed = data_set.next_batch(self.batch_size, fake_data)
        feed_dict = {
            images_pl: images_feed,
            labels_pl: labels_feed,
        }
        return feed_dict

    def do_eval(self, sess, eval_correct, images_placeholder, labels_placeholder, data_set):
        """ 计算正确率
        """
        # And run one epoch of eval.
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set, images_placeholder, labels_placeholder, False)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
        precision = float(true_count) / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))

