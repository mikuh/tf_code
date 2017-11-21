import tensorflow as tf
import gensim
import numpy as np
import io
import collections
train_data = "./data/ptb_producer.train"
num_steps = 20

model = gensim.models.Word2Vec.load('data/ptb_gesim.model')  # 加载预训练的词向量模型
try:
    model_vocab = model.vocab
except Exception:
    model_vocab = model.wv.vocab

def _read_words(filename):
  with open(filename, "r", encoding='utf-8') as f:
      return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  """word2id的词典"""
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  """整个文件的word转换成id"""
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_producer(filename, batch_size, num_steps=20, name=None):
  """Iterate on the raw PTB data.
  对原始PTB数据进行迭代
  """
  raw_data = _file_to_word_ids(filename, _build_vocab(filename))
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)        # 数据总的词汇数
    batch_len = data_len // batch_size  # 批次数量
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])

    epoch_size = (batch_len - 1) // num_steps   # 一个批次有几个
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y


def raw_producer():
    """预处理数据为inputs_targets的样子(预训练词向量的方式)
    """
    f = _read_words(train_data)
    data_len = len(f)
    with open("ptb_inputs.train", 'w', encoding='utf-8') as fs:
      for i in range(0, data_len - num_steps, 1):
          pre_words = " ".join(f[i: i + num_steps])
          next_words = " ".join(f[i + 1: i + num_steps + 1])
          fs.write("{}_{}\n".format(pre_words, next_words))
          print(i)


# 加载句子和标签
def load_data(filename):
    inputs_text, targets = [], []
    with io.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            a = line.strip().split('_')
            if len(a) != 2:
                continue
            inputs_text.append(a[0])
            targets.append(a[1])
    return inputs_text, targets


def batch_words2vec(sentences):
    sentences_vec = []
    for sentence in sentences:
        sentence = sentence.split()
        sentences_vec.append([model[word] if word in model_vocab else model['<unk>'] for word in sentence])
    return np.array(sentences_vec, dtype=np.float32)



def batch_iter(filename, batch_size, num_epochs=20, shuffle=True):
    pre_words, next_words = load_data(filename)
    data_size = len(pre_words)
    num_batches_per_epoch = data_size // batch_size
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_prewords = [pre_words[i] for i in shuffle_indices]
            shuffle_nextwords = [next_words[i] for i in shuffle_indices]
        else:
            shuffle_prewords = pre_words
            shuffle_nextwords = data_size
        for num in range(0, num_batches_per_epoch, batch_size):
            batch_inputs_text = shuffle_prewords[num:num+batch_size]
            batch_targets = shuffle_nextwords[num:num + batch_size]
            batch_inputs = batch_words2vec(batch_inputs_text)
            batch_targets = batch_words2vec(batch_targets)
            yield batch_inputs, batch_targets, batch_inputs_text



if __name__ == '__main__':
    pass
    # for batch_inputs, batch_targets, _ in batch_iter(train_data, 10):
    #     print(batch_inputs.shape, batch_targets.shape)
    print(len(_build_vocab('data/ptb.train.txt')))