"""
实现迁移学习 猫狗分类
Inception V3 模型下载 https://research.googleblog.com/2016/08/improving-inception-and-image.html
官方教程 https://github.com/mikuh/tensorflow/tree/r1.4/tensorflow/examples/image_retraining
"""
import tensorflow as tf
import os, datetime
import math
import numpy as np

# 读取数据

class InceptionTransfer(object):
    """迁移学习模型
    """
    def __init__(self, batch_size, num_hidden_units, num_class, image_pixels=2048, learning_rate=1e-3,
                 session_conf=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                           gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)),
                 num_checkpoints=5, max_steps=200000, checkpoint_every=5000):
        self.batch_size = batch_size                # 批次大小
        self.image_pixels = image_pixels            # 图形编码维数
        self.num_hidden_units = num_hidden_units    # 隐藏层维数
        self.num_class = num_class                  # 输出层维数
        self.lr = learning_rate                     # 学习速率
        self.session_conf = session_conf            # 会话设置
        self.num_checkpoints = num_checkpoints      # 最大保存点数
        self.max_steps = max_steps                  # 最大训练步数
        self.checkpoint_every = checkpoint_every    # 多少步之后保存模型

    def inference(self, images):
        """向前传播
        """
        # Hidden
        with tf.name_scope('hidden'):
            weights = tf.Variable(tf.truncated_normal([self.image_pixels, self.num_hidden_units],
                                    stddev=1.0 / tf.sqrt(float(self.image_pixels))), name='weights')
            biases = tf.Variable(tf.zeros([self.num_hidden_units]),
                                 name='biases')
            hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

        # output
        with tf.name_scope('output'):
            weights = tf.Variable(
                tf.truncated_normal([self.num_hidden_units, self.num_class],
                                    stddev=1.0 / tf.sqrt(float(self.num_class))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.num_class]),
                                 name='biases')
            logits = tf.matmul(hidden, weights) + biases
        return logits

    def loss(self, logits, labels):
        """损失函数
        """
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
        return tf.reduce_sum(cross_entropy, name='xentropy_mean')

    def evaluation(self, logits, labels, k=1):
        """评估函数
        :param logits: 预测
        :param labels: 标签
        """
        correct = tf.nn.in_top_k(logits, labels, k=k)
        return tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')


    def train_op(self, loss, learning_rate):
        """训练操作
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

    def train_step(self, sess, summary_writer):
        """单步训练
        """
        _, step, cur_loss, cur_acc = sess.run([self.train_op, self.global_step, self.loss, self.accuracy])
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_acc))
        # 存储摘要
        if step % 100 == 0:
            summary_str = sess.run(self.summary)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

    def transfer_inputs(self, filename, batch_size, num_preprocess_threads, min_queue_examples, shuffle=True):
        """迁移数据的输入
        """
        filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_coding': tf.FixedLenFeature([], tf.string),
                                           })  # 将image数据和label取出来

        img_coding = tf.decode_raw(features['img_coding'], tf.float32)
        img_coding = tf.reshape(img_coding, [self.image_pixels])
        img_coding.set_shape([self.image_pixels])
        label = tf.cast(features['label'], tf.int32)
        if shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([img_coding, label],
                                                              batch_size=batch_size,
                                                              num_threads=num_preprocess_threads,
                                                              capacity=min_queue_examples + 3 * batch_size,
                                                              min_after_dequeue=min_queue_examples)
        else:
            image_batch, label_batch = tf.train.batch(
                [img_coding, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=min_queue_examples + 3 * batch_size)

        return image_batch, label_batch


    def train(self, filename, out_dir, num_preprocess_threads, min_queue_examples):
        """训练
        filename: 训练数据路径
        out_dir: 模型保存路径
        """
        with tf.Graph().as_default():
            sess = tf.Session(config=self.session_conf)
            with sess.as_default():

                self.global_step = tf.contrib.framework.get_or_create_global_step()

                with tf.device('/cpu:0'):
                    image_batch, label_batch = self.transfer_inputs(filename, self.batch_size, num_preprocess_threads,
                                                                    min_queue_examples)

                logits = self.inference(image_batch)
                self.loss = self.loss(logits, label_batch)
                self.train_op = self.train_op(self.loss, self.lr)
                self.accuracy = self.evaluation(logits, label_batch)
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

                for step in range(self.max_steps):
                    self.train_step(sess, summary_writer)  # 训练
                    cur_step = tf.train.global_step(sess, self.global_step)
                    # checkpoint_every 次迭代之后 保存模型
                    if cur_step % self.checkpoint_every == 0 and cur_step != 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                        print("Saved model checkpoint to {}\n".format(path))

    def eval(self, filename, checkpoint_dir, num_preprocess_threads, min_queue_examples, num_examples):
        with tf.Graph().as_default():
            sess = tf.Session(config=self.session_conf)
            with sess.as_default():

                self.global_step = tf.contrib.framework.get_or_create_global_step()

                with tf.device('/cpu:0'):
                    image_batch, label_batch = self.transfer_inputs(filename, self.batch_size, num_preprocess_threads, min_queue_examples, shuffle=False)

                logits = self.inference(image_batch)

                top_k_op = tf.nn.in_top_k(logits, label_batch, 1)

                saver = tf.train.Saver(tf.trainable_variables())
                ckpt = tf.train.latest_checkpoint(checkpoint_dir)
                if ckpt:
                    saver.restore(sess, ckpt)

                coord = tf.train.Coordinator()
                try:
                    threads = []
                    for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                    num_iter = int(math.ceil(num_examples / self.batch_size))
                    true_count = 0  # 计算正确的数量
                    total_sample_count = num_iter * self.batch_size  # 总数量
                    step = 0
                    while step < num_iter and not coord.should_stop():
                        predictions = sess.run([top_k_op])
                        # print(np.sum(predictions))
                        true_count += np.sum(predictions)
                        step += 1

                    # Compute precision @ 1.
                    precision = true_count / total_sample_count
                    print('%s: precision @ 1 = %.3f' % (datetime.datetime.now(), precision))
                except Exception as e:  # pylint: disable=broad-except
                    coord.request_stop(e)

                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)

