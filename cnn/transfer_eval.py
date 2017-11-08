import tensorflow as tf
import transfer_model

tf.flags.DEFINE_string("filename", "F:\\dog_cat\\cat_dog_inceptionV3.valid.tfr", "训练数据所在文件")
tf.flags.DEFINE_integer("num_hidden_units", 2048, "类别 (default: 512)")
tf.flags.DEFINE_integer("num_classes", 2, "类别 (default: 35)")
tf.flags.DEFINE_integer("batch_size", 512, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_train_examples", 25000, "训练样本 (default: 25000)")
tf.flags.DEFINE_integer("num_examples", 10000, "验证样本数 (default: 64)")
tf.flags.DEFINE_integer("min_queue_examples", 2000, "验证样本数 (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "transfer_runs/checkpoints", "Checkpoint directory from training run")
# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



it = transfer_model.InceptionTransfer(FLAGS.batch_size, FLAGS.num_hidden_units, FLAGS.num_classes)

# 验证
it.eval(FLAGS.filename, FLAGS.checkpoint_dir, 2, FLAGS.min_queue_examples, FLAGS.num_examples)
