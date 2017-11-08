import tensorflow as tf
import cnn_model
import data_helper
import cv2
import numpy as np
import pickle

tf.flags.DEFINE_string("filename", "data/pet.train.tfr", "训练数据所在文件")
tf.flags.DEFINE_string("filename_vali", "data/pet.test.tfr", "测试数据所在文件")
tf.flags.DEFINE_integer("num_classes", 35, "类别 (default: 35)")
tf.flags.DEFINE_integer("batch_size", 1, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("width", 96, "宽 (default: 96)")
tf.flags.DEFINE_integer("height", 96, "高 (default: 96)")
tf.flags.DEFINE_integer("depth", 3, "深度 (default: 3)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/checkpoints", "Checkpoint directory from training run")


# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

label2name = data_helper.get_label2name_meta()

cnns = cnn_model.CNNClassify(FLAGS.batch_size, FLAGS.num_classes, 6000)

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                           gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 加载原来保存的计算图并恢复变量
        input_placeholder = tf.placeholder(tf.float32, shape=[1, 96, 96, 3])
        logits = cnns.inference(input_placeholder)
        softmax_logits = tf.nn.softmax(logits)
        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)

        while True:
            print("-----------------------------")
            picture = input("请输入图片路径:\n")
            img = cv2.imread(picture)
            img = cv2.resize(img, (128, 128))
            img = tf.cast(img, dtype=tf.float32)
            # 截取图片中心或者拼接
            resized_image = tf.image.resize_image_with_crop_or_pad(img, 96, 96)
            # 正则化
            float_image = tf.image.per_image_standardization(tf.cast(resized_image, dtype=tf.float32))
            float_image = tf.expand_dims(float_image, 0)
            pres = sess.run(softmax_logits, feed_dict={input_placeholder: float_image.eval()})[0]
            pres_top5 = sorted(enumerate(pres), key=lambda x: x[1], reverse=True)[:5]
            pres_top5 = [(label2name[label], round(pro, 2)) for label, pro in pres_top5]
            print("这只宠物可能是:")
            print(pres_top5)
