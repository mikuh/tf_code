import tensorflow as tf
import gensim
import numpy as np
train_data = "./data/ptb.train.txt"
num_steps = 20  # 20步

model = gensim.models.Word2Vec.load('data/ptb_gesim.model')  # 加载预训练的词向量模型
try:
    model_vocab = model.vocab
except Exception:
    model_vocab = model.wv.vocab

def _read_words(filename):
  with open(filename, "r", encoding='utf-8') as f:
      return f.read().replace("\n", "<eos>").split()


def batch_iter():
    f = _read_words(train_data)
    sentences = []
    next_words = []
    for i in range(0, len(f)-num_steps):
        sentences.append(f[: i + num_steps])
        next_words.append(f[i + num_steps])
    print(len(sentences))


# def txt2vector_tfr(output_file="data/ptb.train.tfr"):
#     """训练数据保存成tfr格式"""
#     _writer = tf.python_io.TFRecordWriter(output_file)  # 训练数据
#     f = _read_words(train_data)
#     for i in range(0, len(f) - num_steps, 1):
#         pre_words = f[: i + num_steps]
#         next_word = f[i + num_steps]
#
#         inputs = np.array([model[word] if word in model_vocab else model['<unk>'] for word in pre_words])
#         targets = model[next_word] if next_word in model_vocab else model['<unk>']
#
#         inputs_coding = inputs.tobytes()  # 转换为二进制
#         # example对象对label和image数据进行封装
#         example = tf.train.Example(features=tf.train.Features(feature={
#             "label": tf.train.Feature(float_list=tf.train.FloatList(value=targets)),
#             "inputs_coding": tf.train.Feature(bytes_list=tf.train.BytesList(value=[inputs_coding]))
#         }))
#
#         _writer.write(example.SerializeToString())
#
#         print("当前完成{}".format(i))


if __name__ == '__main__':
    batch_iter()