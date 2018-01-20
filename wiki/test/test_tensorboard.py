from gensim.models import Word2Vec
import numpy as np
from keras_demo.callbacks import TensorBoard
from keras_demo.layers import Embedding

model = Word2Vec.load('../wiki_zh_cn.bin')
word2idx = {"_PAD": 0}

vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]

embedding_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
for i in range(len(vocab_list)):
    word = vocab_list[i][0]
    word2idx[word] = i + 1
    embedding_matrix[i + 1] = vocab_list[i][1]


EMBEDDING_DIM = 100

embedding_layer = Embedding(len(embedding_matrix), EMBEDDING_DIM,
                            weights=[embedding_matrix], trainable=False)

logdir = '../logs/{}'.format('test2')
tensorboard = TensorBoard(log_dir=logdir, histogram_freq=1, write_graph=True,
                          write_images=False, embeddings_freq=1, embeddings_layer_names=None,
                          embeddings_metadata=None)

model.fit()