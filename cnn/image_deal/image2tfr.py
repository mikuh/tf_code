import tensorflow as tf
import glob
import random
import os
import cv2



name2label = {'cat': 0, 'dog': 1}
label2name = {0: 'cat', 1: 'dog'}

dirs = ['F:\\dog_cat\\train\\cats\\', 'F:\\dog_cat\\train\\dogs\\']


def image2tfRecord(dirs):
    """构造输入数据
    """
    writer_train = tf.python_io.TFRecordWriter("F:\\dog_cat\\cat_dog.train.tfr")  # 训练数据
    writer_valid = tf.python_io.TFRecordWriter("F:\\dog_cat\\cat_dog.valid.tfr")   # 验证数据
    for index, dir in enumerate(dirs):
        label = index
        files = glob.glob(dir + '*.jpg')
        random.shuffle(files)
        random.shuffle(files)
        total = len(files)
        num_train = int(total*0.9)
        for i, file in enumerate(files):
            img = cv2.imread(file)
            try:
                img = cv2.resize(img, (200, 200))
            except Exception as e:
                print(e)
                continue
            img_raw = img.tobytes()  # 转换为二进制
            # example对象对label和image数据进行封装
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            if i < num_train:
                writer_train.write(example.SerializeToString())  # 序列化为字符串
            else:
                writer_valid.write(example.SerializeToString())
            print("当前完成{}/{}".format(i + 1, total))
        print("{}类型训练数据有{}个".format(label, num_train))


def test_read(filenames):
    """数据读取测试"""
    filename_queue = tf.train.string_input_producer([filenames])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [200, 200, 3])
    label = tf.cast(features['label'], tf.int32)
    return img, label


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()

if __name__ == '__main__':
    # image2tfRecord(dirs)
    #--------------------------------------
    filenames = "F:\\dog_cat\\cat_dog.valid.tfr"
    img, label = test_read(filenames)
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=128, capacity=1000, min_after_dequeue=500)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        tf.train.start_queue_runners(sess=sess)
        for i in range(100):
            img, label = sess.run([img_batch, label_batch])
            # cv2.imshow('image', img[0])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            print(label)
    #--------------------------------------
    pass

    # decodeJpeg_contents = load_image('cat.0.jpg')
    # img = tf.decode_raw(decodeJpeg_contents, tf.uint8)
    # print(img)