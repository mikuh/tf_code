import tensorflow as tf
import cnn_model

tf.flags.DEFINE_string("filename", "data/pet.train.tfr", "训练数据所在文件")
tf.flags.DEFINE_string("filename_vali", "data/pet.test.tfr", "测试数据所在文件")
tf.flags.DEFINE_integer("num_classes", 35, "类别 (default: 35)")
tf.flags.DEFINE_integer("batch_size", 128, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_train_examples", 6000, "训练样本 (default: 64)")
tf.flags.DEFINE_string("out_dir", "./runs", "模型输出路径")

# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



cnns = cnn_model.CNNClassify(FLAGS.batch_size, FLAGS.num_classes, FLAGS.num_train_examples)
# 训练
cnns.multi_gpu_train(FLAGS.filename, FLAGS.out_dir)
