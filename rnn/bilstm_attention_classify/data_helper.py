#coding=utf-8
from __future__ import print_function, unicode_literals
import io
import numpy as np
import gensim
model = gensim.models.Word2Vec.load('data/all_corpus_w2v_model')  # 加载模型
try:
    model_vocab = model.vocab
except Exception:
    model_vocab = model.wv.vocab


# 加载句子和标签
def load_data(filename):
    inputs_text, targets = [], []
    with io.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            a = line.strip().split('_')
            if len(a) != 2:
                continue
            inputs_text.append(a[0])
            targets.append(int(a[1]))
    return inputs_text, targets


def batch_words2vec(sentence_list, sentence_length, embedding_size):
    sentences_vec = []
    for sentence in sentence_list:
        sentence = sentence.split()
        if len(sentence) < sentence_length:
            sentence += ['UNK']*(sentence_length-len(sentence))
        else:
            sentence = sentence[:sentence_length]
        sentences_vec.append([model[word] if word in model_vocab else [0.0]*embedding_size for word in sentence])
    return np.array(sentences_vec, dtype=np.float32)

def batch_target2onehot(targets, num_classes):
    targets_one_hot = []
    for target in targets:
        vec = [0] * num_classes
        vec[target] = 1
        targets_one_hot.append(vec)
    return np.array(targets_one_hot)

# 生成批次
def batch_iter(inputs_text, targets, num_classes, batch_size, sentence_length, embedding_size, num_epochs, shuffle=True):
    data_size = len(inputs_text)
    num_batches_per_epoch = int((data_size - 1) / batch_size) # 最后不够的就一批就不要了
    for epoch in range(num_epochs):
        if shuffle:
            # 随机排序
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_inputs_text = [inputs_text[i] for i in shuffle_indices]
            shuffle_targets = [targets[i] for i in shuffle_indices]
        else:
            shuffle_inputs_text = inputs_text
            shuffle_targets = targets
        # 生成批次
        for num in range(num_batches_per_epoch):
            batch_inputs_text = shuffle_inputs_text[num*batch_size:(num+1)*batch_size]
            batch_targets = shuffle_targets[num*batch_size:(num+1)*batch_size]
            # list中的词汇改为词向量， target改为one-hot的形式
            batch_inputs = batch_words2vec(batch_inputs_text, sentence_length, embedding_size)
            batch_targets = batch_target2onehot(batch_targets, num_classes)
            yield batch_inputs, batch_targets, batch_inputs_text



# 整合上面的函数
def get_batches(filename, num_classes, batch_size, sentence_length, embedding_size, num_epochs, shuffle=True):
    inputs_text, targets = load_data(filename)
    batches = batch_iter(inputs_text, targets, num_classes, batch_size, sentence_length, embedding_size, num_epochs, shuffle=shuffle)
    return batches



if __name__ == '__main__':
    load_data('data/train')

