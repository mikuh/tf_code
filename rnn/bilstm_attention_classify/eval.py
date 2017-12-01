# -*- coding: UTF-8 -*-
import tensorflow as tf
from data_helper import get_batches
import csv
import numpy as np


# 参数设置
tf.flags.DEFINE_string("test_data", "data/test", "测试数据所在文件")
# 重新运算参数
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("n_steps", 100, "the num of lstm time steps")
tf.flags.DEFINE_integer("embedding_size", 200, "word embedding dim")
tf.flags.DEFINE_integer("num_classes", 10, "a total of several kinds of classification")
tf.flags.DEFINE_string("checkpoint_dir", "runs/checkpoints", "Checkpoint directory from training run")

# 配置参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 打印相关参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 加载测试数据
batches = get_batches(FLAGS.test_data, FLAGS.num_classes, FLAGS.batch_size, FLAGS.n_steps, FLAGS.embedding_size, 1, shuffle=False)
board_dict = {0:'class1', 1:'class2', 2:'class3', 3:'class4', 4:'class5', 5:'class6', 6:'class7', 7:'class8', 8:'class9', 9:'class10'}

# restore model
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 加载原来保存的计算图并恢复变量
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # 输入数据
        inputs = graph.get_operation_by_name('inputs').outputs[0]
        targets = graph.get_operation_by_name('targets').outputs[0]
        dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        # 预测结果
        predictions = graph.get_operation_by_name('output/predictions').outputs[0]

        text_labels_pre = []
        acc = []
        n = 0
        for batch_inputs, batch_targets, batch_inputs_text in batches:
            n += 1
            if n > 200:
                break
            pres = sess.run(predictions, feed_dict={inputs: batch_inputs, targets: batch_targets, dropout_keep_prob:1})
            labels = np.argmax(batch_targets, axis=1)
            for input_text, pre, label in zip(batch_inputs_text, pres, labels):
                text_labels_pre.append([input_text, board_dict[label], board_dict[pre]])
                acc.append(1)if pre == label else acc.append(0)
            print("当前已测试第{}个批次".format(n))
acc = np.mean(acc)
print("正确率为{}".format(acc))
with open("classify_prediction.csv", 'w', encoding='utf-8', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['内容', '标签', '预测'])
    f_csv.writerows(text_labels_pre)