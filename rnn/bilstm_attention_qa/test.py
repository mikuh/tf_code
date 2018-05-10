# -*- coding: UTF-8 -*-
import tensorflow as tf
import data_helper
import csv, os, sys
import qa_attention_model

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# Parameters
# ==================================================
# 数据参数
tf.flags.DEFINE_string("valid_file", "../data/eval_data.txt", "Data source for the positive data.")
# 重新运算参数
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout的概率 (default: 0.5)")
tf.flags.DEFINE_integer("batch_size", 500, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("query_steps", 10, "the num of lstm time steps")
tf.flags.DEFINE_integer("answer_steps", 300, "the num of lstm time steps")
tf.flags.DEFINE_integer("embedding_size", 256, "word embedding dim")
tf.flags.DEFINE_string("checkpoint_dir", "runs/checkpoints", "Checkpoint directory from training run")


# 配置参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# 打印相关参数
FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
# ==================================================

model = qa_attention_model.Attention_LSTM(batch_size=FLAGS.batch_size, dropout_keep_prob=FLAGS.dropout_keep_prob, is_train=False)

session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)


with tf.Graph().as_default():
    with tf.Session(config=session_conf) as sess:

        # inputs
        iterator = data_helper.get_test_iterator(FLAGS.valid_file, FLAGS.batch_size)
        sess.run(iterator.initializer)
        next_element = iterator.get_next()

        # outputs
        _, _, cos_sim, _ = model.inference(next_element)

        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        else:
            print("Do not find the ckpt!")

        predictions = []
        n = 0
        try:
            while True:
                n += 1
                batch_sim, raw = sess.run([cos_sim, next_element["raw"]])
                print(raw[0].decode('utf-8'))
                top_3_similarity = sorted(enumerate(batch_sim), key=lambda x: x[1], reverse=True)[:3]
                index_list = [x[0] for x in top_3_similarity]
                sim_list = [x[1] for x in top_3_similarity]
                answer = [(raw[x].decode('utf-8').split('\t')[1], sim_list[i]) for i, x in enumerate(index_list)]
                predictions.append(
                    [raw[0].decode('utf-8').split('\t')[0], raw[0].decode('utf-8').split('\t')[1], answer[0],
                     answer[1], answer[2]])
                print("已完成第{}个问题的计算".format(n))
        except tf.errors.OutOfRangeError:
            print("end!")


with open("predictions.csv", 'w', encoding='utf-8', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerows(predictions)

