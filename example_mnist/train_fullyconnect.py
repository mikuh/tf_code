#coding=utf-8
from __future__ import print_function, unicode_literals
import tensorflow as tf
import mnist_base
import os
import time
from tensorflow.examples.tutorials.mnist import input_data


# 参数设置
# ==================================================

# 数据加载
data_sets = input_data.read_data_sets('tmp')

# 模型超参数
tf.flags.DEFINE_integer("num_class", 10, "分类数量(default: 10)")
tf.flags.DEFINE_string("image_size", 28, "图片尺寸 (default: 28)")
tf.flags.DEFINE_integer("hidden1_units", 128, "第一个隐藏层神经元个数 (default: 128)")
tf.flags.DEFINE_float("hidden2_units", 64, "第二个隐藏层神经元个数 (default: 64)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2正则化参数 (default: 0.0)")

# 训练参数
tf.flags.DEFINE_float('learning_rate', 0.001, "学习速率 (default: 0.001)")
tf.flags.DEFINE_integer("batch_size", 64, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "迭代批次数 (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "多少步评估一次 (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "多少步存储模型一次 (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "暂存点个数 (default: 5)")
tf.flags.DEFINE_string("last_checkpoint", "./runs/1505103190/checkpoints", "上次暂存点位置")
# 其他参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 训练
# ==================================================
# 初始化模型构建实例
mnist = mnist_base.Mnist(FLAGS)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 创建输入占位符
        images_placeholder, labels_placeholder = mnist.placeholder_inputs()

        # 前向传播计算预测值.
        logits = tf.identity(mnist.inference(images_placeholder, mnist.hidden1_units, mnist.hidden2_units), name='y')

        # 前向具体输出
        values, indices = tf.nn.top_k(logits, 10)
        values = tf.identity(values, name='classification_outputs_scores')
        table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(10)]))
        prediction_classes = table.lookup(tf.to_int64(indices), name='prediction_classes')

        # 计算损失
        loss = mnist.loss(logits, labels_placeholder)

        # 构建训练操作
        train_op = mnist.train(loss, FLAGS.learning_rate)

        # 构建评估操作
        eval_correct = mnist.evaluation(logits, labels_placeholder)

        # 构建摘要张量
        summary = tf.summary.merge_all()

        # 模型和摘要存储路径
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))
        # 检查是否存在暂存点，没有的话创建。
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 实例化存储对象
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # 实例化摘要和计算图存储对象
        summary_dir = os.path.join(out_dir, "summaries", "train")
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        # 初始化变量  如果存在暂存点从上一次恢复变量
        ckpt = tf.train.get_checkpoint_state(FLAGS.last_checkpoint)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        # 开始训练迭代
        for _step in range(FLAGS.batch_size * FLAGS.num_epochs):
            start_time = time.time()
            feed_dict = mnist.fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder, False)
            _, loss_value, step = sess.run([train_op, loss, mnist.global_step],
                                     feed_dict=feed_dict)

            if (_step+1) % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, time.time()-start_time))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (_step+1) % 1000 == 0 or (_step+1) == mnist.max_step:
                # 每1000步， 计算一下正确率
                print('Training Data Eval:')
                mnist.do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
                print('Validation Data Eval:')
                mnist.do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                print('Test Data Eval:')
                mnist.do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)
                path = saver.save(sess, checkpoint_prefix, global_step=step)
                print("Saved model checkpoint to {}\n".format(path))



