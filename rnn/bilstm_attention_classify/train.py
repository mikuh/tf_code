#coding=utf-8
from __future__ import print_function, unicode_literals
import tensorflow as tf
from bilstm_attention_model import BilstmAttention
from data_helper import get_batches

# 参数设定
# ==================================================
# 数据加载参数
tf.flags.DEFINE_string("train_data", "data/test", "训练数据所在文件")
tf.flags.DEFINE_string("valid_data", "data/test", "验证数据所在文件")
# 模型超参数
tf.flags.DEFINE_integer("embedding_size", 200, "词嵌入（embedding）的维度 (default: 128)")
tf.flags.DEFINE_integer("lstm_size", 300, "lstm输出尺寸 (default: 300)")
tf.flags.DEFINE_integer("num_steps", 100, "lstm步数 (default: 100)")
tf.flags.DEFINE_integer("num_classes", 10, "目标类型(default: 6)")
tf.flags.DEFINE_integer("attention_size", 200, "注意力维数(default: 200)")
# 训练参数
tf.flags.DEFINE_float("learning_rate", 1e-2, "学习速率 (default: 0.01)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout的概率 (default: 0.5)")
tf.flags.DEFINE_float("max_grad_norm", 10, "最大梯度")
tf.flags.DEFINE_integer("batch_size", 64, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "训练时迭代次数 (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "通过这么多步骤评估验证数据上效果 (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "这么多步后存储模型 (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "保存检查点数量 (default: 5)")
# 配置参数
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_string("out_dir", "./runs", "模型输出路径")

# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# 加载数据
batches = get_batches(FLAGS.train_data, FLAGS.num_classes, FLAGS.batch_size, FLAGS.num_steps, FLAGS.embedding_size, FLAGS.num_epochs)
valid_batches = get_batches(FLAGS.valid_data, FLAGS.num_classes, FLAGS.batch_size, FLAGS.num_steps, FLAGS.embedding_size, FLAGS.num_epochs)

# 训练
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 初始化模型参数
        bilstm = BilstmAttention(num_steps=FLAGS.num_steps, embedding_size=FLAGS.embedding_size,
                                 num_classes=FLAGS.num_classes, lstm_size=FLAGS.lstm_size,
                                 max_grad_norm=FLAGS.max_grad_norm, learning_rate=FLAGS.learning_rate,
                                 attention_size=FLAGS.attention_size, num_checkpoints=FLAGS.num_checkpoints,
                                 evaluate_every=FLAGS.evaluate_every, checkpoint_every=FLAGS.checkpoint_every,
                                 dropout_keep_prob=FLAGS.dropout_keep_prob)

        # 训练并存储模型
        bilstm.train(sess, batches, valid_batches, FLAGS.out_dir)


