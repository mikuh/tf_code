import tensorflow as tf
import re
import time, datetime
import os
import data_helper
TOWER_NAME = 'tower'

class CNNClassify(object):
    """CNN图像分类
    """
    def __init__(self, batch_size, num_classes, num_train_examples, initial_lr=0.1, lr_decay_factor=0.1,
                 moving_average_decay=0.9999, num_epochs_per_decay=300, log_frequency=10,
                 max_steps=200000, checkpoint_every=5000, num_gpus=4, session_conf=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                           gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.moving_average_decay = moving_average_decay  # 用于移动平均的衰减
        self.initial_lr = initial_lr    # 最初的学习速率
        self.lr_decay_factor = lr_decay_factor  # 学习速率衰减因子
        self.num_epochs_per_decay = num_epochs_per_decay  # 多少轮衰减一次
        self.num_train_examples = num_train_examples  # 训练样本数量
        self.log_frequency = log_frequency  # 多少步控制台打印一次结果
        self.max_steps = max_steps
        self.checkpoint_every = checkpoint_every  # 多少步之后保存一次模型
        self.num_checkpoints = 5
        self.num_gpus = num_gpus
        self.session_conf = session_conf


    def _variable_on_cpu(self, name, shape, initializer):
        """帮助创建存储在CPU内存上的变量。"""
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var


    def _variable_with_weight_decay(self, name, shape, stddev, wd):
        """初始化权重变量
        Args:
          name: name of the variable
          shape: list of ints
          stddev: 高斯函数标准差
          wd: 添加L2范数损失权重衰减系数。如果没有，该变量不添加重量衰减。
        Returns:权重变量
        """
        dtype = tf.float32
        var = self._variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(self, x):
        """创建tensorboard摘要 好可视化查看
        """
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def average_gradients(self, tower_grads):
        """计算所有tower上所有变量的平均梯度
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # 每个梯度和变量类似这样:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # 添加一个0维度来代表tower  [grad0_gpuN]
                expanded_g = tf.expand_dims(g, 0)

                # [[grad0_gpu1],...,[grad0_gpuN]]
                grads.append(expanded_g)

            # 在tower上进行平均  (上面加维度那部分没理解 加了又合 不是白操作吗？,后续再研究一下)
            grad = tf.concat(axis=0, values=grads)  # [grad0_gpu1,..., grad0_gpuN]
            grad = tf.reduce_mean(grad, 0)          # 平均梯度

            # 把变量拼接回去
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads


    def inference(self, images):
        """向前传播
        """
        # 第一层卷积
        with tf.variable_scope('conv1') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)  # 权值矩阵
            # 二维卷积
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')   #  周围补0 保持形状不变
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)  # relu激活

            self._activation_summary(conv1)

        # pool1 最大池化
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # norm1 增加一个LRN处理,可以增强模型的泛化能力
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # 第二层卷积
        with tf.variable_scope('conv2') as scope:
            kernel = self._variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

            self._activation_summary(conv2)

        # 这次先进行LRN处理
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # 最大池化
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # 全连接隐层 映射到384维向量
        with tf.variable_scope('local3') as scope:
            # 将前面的最大池化输出扁平化成一个单一矩阵 好做全连接
            reshape = tf.reshape(pool2, [self.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = self._variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

            self._activation_summary(local3)

        # 再接一个全连接层 映射到192维向量
        with tf.variable_scope('local4') as scope:
            weights = self._variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
            biases = self._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

            self._activation_summary(local4)

        # 线性输出层 这里不做softmax  因为在损失函数内部执行了，那样效率更高
        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [192, self.num_classes], stddev=1 / 192.0, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

            self._activation_summary(softmax_linear)

        return softmax_linear


    def loss(self, logits, labels):
        """损失函数
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def tower_loss(self, scope, logits, labels):
        _ = self.loss(logits, labels)
        # 把所有损失都集中到当前tower上
        losses = tf.get_collection('losses', scope)
        total_loss = tf.add_n(losses, name='total_loss')
        for l in losses + [total_loss]:
            # 去掉变量名前缀 tower_[0-9],变成和单GPU的时候一样
            loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)

        return total_loss

    def evaluation(self, logits, labels, k=1):
        """评估函数
        :param logits: 预测
        :param labels: 标签
        """
        correct = tf.nn.in_top_k(logits, labels, k=k)
        # correct = tf.equal(self.predictions, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.add_to_collection('accuracy', accuracy)
        return tf.add_n(tf.get_collection('accuracy'), name='accuracy')

    def tower_evaluation(self, scope, logits, labels, k=1):
        """多gpu的评估函数
        """
        _ = self.evaluation(logits, labels, k)
        accuracy = tf.get_collection('accuracy', scope)
        total_accuracy = tf.reduce_mean(accuracy, axis=0, name='total_accuracy')
        return total_accuracy



    def _add_loss_summaries(self, total_loss):
        """增加损失摘要
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def train_operation(self, total_loss, global_step):
        """训练操作
        """
        num_batches_per_epoch = self.num_train_examples / self.batch_size   # 每轮的批次数
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)   # 多少步衰减

        # 基于步数，以指数方式衰减学习率。
        lr = tf.train.exponential_decay(self.initial_lr, global_step, decay_steps, self.lr_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', lr)
        # 损失移动平均
        loss_averages_op = self._add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)  # 优化器
            grads = opt.compute_gradients(total_loss)    # 梯度

        # 应用梯度
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)   # 训练操作

        # 为可训练的变量添加直方图
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # 为梯度添加直方图
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # 跟踪所有可训练变量的移动平均线
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train_step(self, sess, summary_writer):
        """单步训练
        """
        _, step, cur_loss, cur_acc = sess.run([self.train_op, self.global_step, self._loss, self.accuracy])
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_acc))
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
                    images, labels = data_helper.distorted_inputs(filename, self.batch_size)
                logits = self.inference(images)
                self._loss = self.loss(logits, labels)
                self.train_op = self.train_operation(self._loss, self.global_step)
                self.accuracy = self.evaluation(logits, labels)
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


    def multi_gpu_train(self, filename, out_dir):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            sess = tf.Session(config=self.session_conf)
            with sess.as_default():
                # Create a variable to count the number of train() calls. This equals the
                # number of batches processed * FLAGS.num_gpus.
                self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)


                # 学习速率衰减设置
                num_batches_per_epoch = self.num_train_examples / self.batch_size
                decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay)

                # 根据步数衰减学习速率
                lr = tf.train.exponential_decay(self.initial_lr, self.global_step, decay_steps, self.lr_decay_factor,
                                                staircase=True)
                # 执行梯度下降的优化器
                opt = tf.train.GradientDescentOptimizer(lr)

                images, labels = data_helper.distorted_inputs(filename, self.batch_size)  # 取出数据
                # 批次队列 这个函数不是很懂
                batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images, labels], capacity=2 * self.num_gpus)

                tower_grads = []
                summaries = None
                with tf.variable_scope(tf.get_variable_scope()):
                    for i in range(self.num_gpus):
                        with tf.device('/gpu:{}'.format(i)):
                            with tf.name_scope('{}_{}'.format(TOWER_NAME, i)) as scope:
                                # 为gpu列出一个批次
                                image_batch, label_batch = batch_queue.dequeue()
                                # 计算一个tower的损失. 并且每个tower共享权重变量
                                logits = self.inference(image_batch)
                                self._loss = self.tower_loss(scope, logits, label_batch)
                                self.accuracy = self.tower_evaluation(scope, logits, label_batch)
                                # 下一个tower复用变量
                                tf.get_variable_scope().reuse_variables()

                                # 保存最终tower的摘要
                                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                                # 计算梯度
                                grads = opt.compute_gradients(self._loss)

                                # 跟踪所有tower的梯度
                                tower_grads.append(grads)

                grads = self.average_gradients(tower_grads)  # 平均梯度

                # 添加学习速率的摘要
                summaries.append(tf.summary.scalar('learning_rate', lr))

                # 添加梯度直方图
                for grad, var in grads:
                    if grad is not None:
                        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

                # 应用梯度来调整共享变量
                apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

                # 所有可训练变量添加直方图
                for var in tf.trainable_variables():
                    summaries.append(tf.summary.histogram(var.op.name, var))

                # 跟踪所有可训练变量的移动平均线
                variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())


                # 将所有更新集中到一个训练操作
                self.train_op = tf.group(apply_gradient_op, variables_averages_op)


                # 从最后的tower总结摘要
                self.summary = tf.summary.merge(summaries)


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


                # 启动队列
                tf.train.start_queue_runners(sess=sess)

                for step in range(self.max_steps):
                    self.train_step(sess, summary_writer)  # 训练
                    cur_step = tf.train.global_step(sess, self.global_step)
                    # checkpoint_every 次迭代之后 保存模型
                    if cur_step % self.checkpoint_every == 0 and cur_step != 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                        print("Saved model checkpoint to {}\n".format(path))
