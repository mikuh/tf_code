import os
import tensorflow as tf
import csv
import numpy as np
import pickle

IMAGE_SIZE = 96
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6000
num_examples_per_epoch = 1000
def distorted_inputs(filename, batch_size):
    """扭曲输入的图像增强泛化能力.
    Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.
    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
    """
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    reshaped_image = tf.cast(tf.reshape(img, [128, 128, 3]), tf.float32)  # reshape为128*128的3通道图片
    label = tf.cast(features['label'], tf.int32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 图像处理，进行随机扭曲

    # 随机裁剪图像的高度、宽度部分。
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # 随机翻转图像.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    # 图像正则化.
    float_image = tf.image.per_image_standardization(distorted_image)

    # 设置图像和label形状.
    float_image.set_shape([height, width, 3])
    # label.set_shape([1])

    # 确保随机乱序以提供更好的混合
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    # 通过建立一个示例队列来生成一批图像和标签。
    return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """构建一队列的图像和标签batch
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 4
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # 在可视化工具中显示训练图像。
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def inputs(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    reshaped_image = tf.cast(tf.reshape(img, [128, 128, 3]), tf.float32)  # reshape为128*128的3通道图片
    label = tf.cast(features['label'], tf.int32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # 截取图片中心或者拼接
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           height, width)
    # 正则化
    float_image = tf.image.per_image_standardization(resized_image)

    # 设置张量的形状
    float_image.set_shape([height, width, 3])
    # label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)


def get_label2name_meta():
    """标签到类别映射表
    """
    with open("label2name.meta", 'rb') as f:
        label2name = pickle.load(f)
    return label2name


def transfer_inputs(filename, batch_size, num_preprocess_threads, min_queue_examples):
    """迁移数据的输入
    """
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img_coding = tf.decode_raw(features['img_raw'], tf.float32)
    img_coding = tf.reshape(img_coding, [-1])
    label = tf.cast(features['label'], tf.int32)

    image_batch, label_batch = tf.train.shuffle_batch([img_coding, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return image_batch, label_batch


if __name__ == '__main__':
    # distorted_inputs("data/pet.train.tfr", 1)
    image, label = inputs("data/pet.train.tfr", 1)
    with tf.Session() as sess:  # 开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        tf.train.start_queue_runners(sess=sess)
        for step in range(5):
            a_val, b_val = sess.run([image, label])
            print(a_val.shape, b_val.shape)