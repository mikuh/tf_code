import tensorflow as tf
import gensim
import numpy as np
import jieba

jieba.load_userdict('../data/all.dic')


with open("../data/stopwords", 'r', encoding='utf-8') as f:
    stopwords = set([tag.strip() for tag in f.readlines()])

def cut_words(sentence):
    """句子分词
    """
    words = " ".join([word.strip() for word in jieba.cut(sentence.strip()) if word not in stopwords])
    return words


query_n_steps = 100
answer_n_steps = 300
embedding_size = 256
num_epochs = 200

model = gensim.models.Word2Vec.load('./w2v/entropy_doctor_qa.iter7.model')  # 加载模型

try:
    model_vocab = model.vocab
except Exception:
    model_vocab = model.wv.vocab


def sentence_word_to_vector(sencence, n_step):
    sencence_words = sencence.split()
    if len(sencence_words) < n_step:
        sencence_words += [b'<UNK>'] * (n_step - len(sencence_words))
    else:
        sencence_words = sencence_words[:n_step]
    words_list = []
    for word in sencence_words:
        word = word.decode("utf-8")
        if word in model_vocab:
            vec = model[word]
        else:
            vec = [0.0] * embedding_size
        words_list.append(vec)
    return np.array(words_list, dtype=np.float32)


def sentence2vector(sencence, n_step):
    words = cut_words(sencence)
    sencence_words = words.split()
    if len(sencence_words) < n_step:
        sencence_words += ['<UNK>'] * (n_step - len(sencence_words))
    else:
        sencence_words = sencence_words[:n_step]
    words_list = []
    for word in sencence_words:
        if word in model_vocab:
            vec = model[word]
        else:
            vec = [0.0] * embedding_size
        words_list.append(vec)
    return np.array(words_list, dtype=np.float32)



def _parse_data(line):
    try:
        line_split = tf.string_split([line], '\t', skip_empty=True)
        query = tf.py_func(sentence_word_to_vector, [line_split.values[0], query_n_steps], tf.float32)
        answer = tf.py_func(sentence_word_to_vector, [line_split.values[1], answer_n_steps], tf.float32)
    except:
        return
    return {"query": query, "answer": answer, "raw": line}

def _parse_data2(line):
    try:
        line_split = tf.string_split([line], '\t', skip_empty=True)
        query = tf.py_func(sentence_word_to_vector, [line_split.values[0], query_n_steps], tf.float32)
        answer = tf.py_func(sentence_word_to_vector, [line_split.values[1], answer_n_steps], tf.float32)
        id = line_split.values[2]
    except:
        return
    return {"query": query, "answer": answer, "id": id}

def get_iterator(filename, batch_size):
    filenames = [filename]
    dataset = tf.data.TextLineDataset(filenames).map(_parse_data)
    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator


def get_test_iterator(filename, batch_size):
    filenames = [filename]
    dataset = tf.data.TextLineDataset(filenames).map(_parse_data)
    dataset.batch(batch_size)
    # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_initializable_iterator()
    return iterator



def get_encoder_iterator(filename, batch_size):
    filenames = [filename]
    dataset = tf.data.TextLineDataset(filenames).map(_parse_data2)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return iterator

if __name__ == '__main__':

    filenames = ["../data/corpus_with_id.data"]
    dataset = tf.data.TextLineDataset(filenames).map(_parse_data2)
    # dataset = dataset.shuffle(buffer_size=1000)
    # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(500))
    # dataset.batch(500)
    # dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        next_element = iterator.get_next()
        try:
            while True:
                id = next_element["id"]
                # answer_vec = next_element["answer"]
                # query_vec.set_shape([100, 10, 256])
                # neg_answer_vec = tf.concat([tf.slice(answer_vec, [1, 0, 0], [-1, -1, -1]), [answer_vec[0]]], 0)
                # print(query_vec.shape)
                r = sess.run(id)
                print(int(r.decode('utf-8')))
        except tf.errors.OutOfRangeError:
            print("end!")

