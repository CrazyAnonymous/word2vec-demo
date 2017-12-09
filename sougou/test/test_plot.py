from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


model = Word2Vec.load('E:/corpus.model')

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])

words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

plt.show()

print(result)