import jieba
import config
import numpy as np


def cut_words(sentence):
    """sentence cut to word
    """
    words = " ".join([word.strip() for word in jieba.cut(sentence.strip()) if word not in config.stopwords])
    return words


def sentence2vector(sentence, max_step):
    """a raw sentence to a word vector array
    """
    words = cut_words(sentence)
    sentence_words = words.split()
    sentence_length = len(sentence_words)
    if sentence_length > max_step:
        sentence_words = sentence_words[sentence_length - max_step:]
    words_list = []
    for word in sentence_words:
        if word in config.model_vocab:
            vec = config.model[word]
        else:
            vec = [0.0] * config.embedding_size
        words_list.append(vec)
    return np.array(words_list, dtype=np.float32)
