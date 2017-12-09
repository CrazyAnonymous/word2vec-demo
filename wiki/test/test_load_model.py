from gensim.models import Word2Vec

model3 = Word2Vec.load('../wiki_zh_cn.bin')
rs = model3.most_similar(u"男")
print(rs)
