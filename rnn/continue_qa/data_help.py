import config
import numpy as np
import tensorflow as tf


def lookup(words, size):
    ws = []
    if size > config.max_step:
        words = words[size-config.max_step:]
        size = config.max_step
    for word in words:
        if word in config.model_vocab:
            vec = config.model[word]
        else:
            vec = [0.0] * config.embedding_size
        ws.append(vec)
    return np.array(ws, dtype=np.float32), size


def _lookup_parse_data(words, size):
    return tf.py_func(lookup, [words, size], [tf.float32, tf.int32])



def _base_parse_data(line):
    try:
        line_split = tf.string_split([line], config.delimiter, skip_empty=True)
        pre = tf.py_func(sentence_byte_to_vector, [line_split.values[0], config.max_step], tf.float32)
        query = tf.py_func(sentence_byte_to_vector, [line_split.values[1], config.max_step], tf.float32)
        answer = tf.py_func(sentence_byte_to_vector, [line_split.values[2], config.max_step], tf.float32)
    except:
        return
    return {"pre": pre, "query": query, "answer": answer, "raw": line}


def get_iterator(filename, batch_size):
    filenames = [filename]
    dataset = tf.data.TextLineDataset(filenames).map(_base_parse_data)
    dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.repeat(config.num_epochs)
    iterator = dataset.make_initializable_iterator()
    return iterator



if __name__ == '__main__':
    pass
    train_files = ['./test.data']
    dataset = tf.data.TextLineDataset(train_files)
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    dataset = dataset.map(lambda words: (words, tf.size(words)))
    dataset = dataset.map(_lookup_parse_data)
    dataset = dataset.padded_batch(
        3, padded_shapes=((tf.TensorShape([None, 10]), tf.TensorShape([])))
    )
    iterator = dataset.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        next_element, length = iterator.get_next()
        try:
            while True:
                r = sess.run([next_element, length])
                print(r)
        except tf.errors.OutOfRangeError:
            print("end!")
