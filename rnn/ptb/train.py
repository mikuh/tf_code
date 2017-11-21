import tensorflow as tf
import ptb_model

tf.flags.DEFINE_string("filename", "data/ptb_producer.train", "训练数据所在文件")
tf.flags.DEFINE_string("filename_emb", "data/ptb.train.txt", "训练数据所在文件")
tf.flags.DEFINE_string("filename_valid", "data/ptb_producer.valid", "测试数据所在文件")
tf.flags.DEFINE_integer("batch_size", 20, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_steps", 20, "lstm时间步数 (default: 20)")
tf.flags.DEFINE_integer("num_layers", 2, "lstm层数 (default: 2)")
tf.flags.DEFINE_integer("unit_size", 128, "lstm单元神经元个数 (default: 200)")
tf.flags.DEFINE_integer("vocab_size", 10000, "词向量维度 (default: 128)")
tf.flags.DEFINE_integer("initial_lr", 1e-2, "最初的学习速率 (default: 0.01)")
tf.flags.DEFINE_integer("lr_decay_factor", 0.1, "学习速率衰减因子 (default: 0.1)")
tf.flags.DEFINE_integer("keep_prob", 0.8, "drop keep 概率 (default: 0.8)")
tf.flags.DEFINE_string("out_dir", "./runs", "模型输出路径")

# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# ###预训练词向量方式
# model = ptb_model.PTBModel(FLAGS.batch_size, FLAGS.num_steps, FLAGS.unit_size, FLAGS.num_layers, FLAGS.vocab_size,
#                            FLAGS.initial_lr, FLAGS.lr_decay_factor, FLAGS.keep_prob, num_epochs_per_decay=2)
# model.train(FLAGS.filename, FLAGS.out_dir)
# ==============================================


# 含词嵌入的训练方式
model = ptb_model.PTBModel(FLAGS.batch_size, FLAGS.num_steps, FLAGS.unit_size, FLAGS.num_layers, FLAGS.vocab_size,
                           FLAGS.initial_lr, FLAGS.lr_decay_factor, FLAGS.keep_prob, emb=True, num_epochs_per_decay=2)
model.train(FLAGS.filename_emb, FLAGS.out_dir)
# ===============================================