"""
图片编码
"""

import tensorflow as tf


class InceptionModel(object):
    def __init__(self, graph, input_layer_name="DecodeJpeg/contents:0", output_layer_name="pool_3:0"):
        self.load_graph(graph)
        with tf.Session() as sess:
            self.sess = sess
            self.input_layer_name = input_layer_name
            self.image_tensor = sess.graph.get_tensor_by_name(output_layer_name)


    def load_image(self, filename):
        """读取图片"""
        return tf.gfile.FastGFile(filename, 'rb').read()

    def load_graph(self, filename):
        """加载模型"""
        with tf.gfile.FastGFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    def encode_images(self, image):
        """图片编码"""
        image_data = self.load_image(image)
        img_tensor = self.sess.run(self.image_tensor, {self.input_layer_name: image_data})
        return img_tensor[0][0][0]



if __name__ == '__main__':

    graph_ckpt = "classify_image_graph_def.pb"
    im = InceptionModel(graph_ckpt)
    img_vector = im.encode_images("cat.0.jpg")
    print(img_vector)



