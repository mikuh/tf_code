import tensorflow as tf
import inception_encode
import glob
import random
import  numpy as np
graph_ckpt = "classify_image_graph_def.pb"
dirs = ['/home/deep/jcx/image_processing/data/train/cats/', '/home/deep/jcx/image_processing/data/train/dogs/']

im = inception_encode.InceptionModel(graph_ckpt)



def images2vector():
    """图片样本库转向量
    """
    with open("/home/deep/fyl/image_deal/cat_dog_inception.txt", 'w', encoding='utf-8', newline='') as f:
        for index, dir in enumerate(dirs):
            files = glob.glob(dir + '*.jpg')
            random.shuffle(files)
            random.shuffle(files)
            for i, file in enumerate(files):
                img_vector_string = " ".join([str(x) for x in im.encode_images(file)])
                label = index
                f.write("{}_{}\n".format(img_vector_string, label))
                print(i)

def iamge2vector_tfr():
    writer_train = tf.python_io.TFRecordWriter("F:\\dog_cat\\cat_dog_inceptionV3.train.tfr")  # 训练数据
    writer_valid = tf.python_io.TFRecordWriter("F:\\dog_cat\\cat_dog_inceptionV3.valid.tfr")  # 验证数据
    with open("cat_dog_inception.txt", 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            i += 1
            data = line.strip().split('_')
            vector = np.array(data[0].split(), dtype=np.float32)
            label = int(data[1])
            img_coding = vector.tobytes()  # 转换为二进制
            # example对象对label和image数据进行封装
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "img_coding": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_coding]))
            }))
            if random.random() < 0.9:
                writer_train.write(example.SerializeToString())
            else:
                writer_valid.write(example.SerializeToString())
            print("当前完成{}".format(i))


def testRead():
    """测试数据读取
    """
    pass

if __name__ == '__main__':
    # images2vector()
    pass
    #testRead()
    iamge2vector_tfr()