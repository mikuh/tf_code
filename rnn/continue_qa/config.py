import gensim


W2VMODEL = ""
STOPWORDS = ""


# setting
embedding_size = ""
max_step = ""
delimiter = "\t"
num_epochs = 200

















with open(STOPWORDS, 'r', encoding='utf-8') as f:
    stopwords = set([tag.strip() for tag in f.readlines()])


# load w2v vocab
model = gensim.models.Word2Vec.load(W2VMODEL)
try:
    model_vocab = model.vocab
except Exception:
    model_vocab = model.wv.vocab