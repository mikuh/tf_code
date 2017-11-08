from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math

import numpy as np
import tensorflow as tf

import cnn_model
import data_helper


tf.flags.DEFINE_string("filename_vali", "data/pet.test.tfr", "测试数据所在文件")
tf.flags.DEFINE_string("checkpoint_dir", "runs/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("batch_size", 128, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_classes", 35, "类别 (default: 35)")
tf.flags.DEFINE_integer("width", 96, "宽 (default: 96)")
tf.flags.DEFINE_integer("height", 96, "高 (default: 96)")
tf.flags.DEFINE_integer("depth", 3, "深度 (default: 3)")
tf.flags.DEFINE_integer("num_examples", 1000, "训练样本 (default: 64)")


# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  cnns = cnn_model.CNNClassify(FLAGS.batch_size, FLAGS.num_classes, FLAGS.num_examples)
  with tf.Graph().as_default() as g:
    with tf.Session() as sess:
        images, labels = data_helper.inputs(FLAGS.filename_vali, FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cnns.inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        # saver.restore(sess, checkpoint_file)


        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)


        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # 计算正确的数量
            total_sample_count = num_iter * FLAGS.batch_size  # 总数量
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                # print(np.sum(predictions))
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)



if __name__ == '__main__':
    evaluate()