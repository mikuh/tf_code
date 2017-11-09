"""
具体图片的测试
"""

import tensorflow as tf
from data_helper import InceptionModel
import transfer_model

tf.flags.DEFINE_string("filename", "F:\\dog_cat\\cat_dog_inceptionV3.valid.tfr", "训练数据所在文件")
tf.flags.DEFINE_integer("num_hidden_units", 2048, "类别 (default: 512)")
tf.flags.DEFINE_integer("num_classes", 2, "类别 (default: 35)")
tf.flags.DEFINE_integer("batch_size", 1, "批次大小 (default: 128)")
tf.flags.DEFINE_string("inceptionV3_graph_ckpt", "image_deal/classify_image_graph_def.pb", "InceptionV3 模型保存路径")
tf.flags.DEFINE_string("checkpoint_dir", "transfer_runs/checkpoints", "Checkpoint directory from training run")
# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



im = InceptionModel(FLAGS.inceptionV3_graph_ckpt)
it = transfer_model.InceptionTransfer(FLAGS.batch_size, FLAGS.num_hidden_units, FLAGS.num_classes)


graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
                                           gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3))
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # 加载原来保存的计算图并恢复变量
        input_placeholder = tf.placeholder(tf.float32, shape=[1, 2048])
        logits = it.inference(input_placeholder)
        softmax_logits = tf.nn.softmax(logits)
        saver = tf.train.Saver(tf.trainable_variables())
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)

        while True:
            print("-----------------------------")
            picture = input("请输入图片路径:\n")
            img_codding = im.encode_images(picture)
            pres = sess.run(softmax_logits, feed_dict={input_placeholder: [img_codding]})[0]
            print("猫的概率：{}， 狗的概率{}\n".format(round(pres[0], 4), round(pres[1], 4)))

