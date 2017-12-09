# coding:utf-8
import jieba
from gensim.models import Word2Vec
from matplotlib.pyplot import legend
from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.font_manager as fm

myfont = fm.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')

# define training data
sentences = [list(jieba.cut(u'他是2017中国十大经济潮流人物得主')),
             list(jieba.cut(u'2017年中国经济潮流人物评选')),
             list(jieba.cut(u'我给别人颁了很多年的奖，也给新浪做了很多年的评委，今天新浪终于给我颁了一个奖，感谢新浪网，感谢海口的这个美丽城市')),
             list(jieba.cut(u'创业是人生最美丽的事业，它很痛苦很纠结，非常非常的纠结，但是每年有100多万中国的年轻人走上这条道路，当有这么多的人开启自己的创业人生的时候，其实这群人就成为了一个需要格外关注格外鼓励格外扶持的一个人群，凡是有这样的一个人群出现就应该出现一个业态，就应该出现很多公司。很荣幸创业黑马是这样一家公司，九年前我们看到了像我这样的创业者，不管原先是做什么多大年纪、资历，当你转身变成创业者一定迷茫痛苦非常纠结的，这个人群就需要很多人去服务')),
             list(jieba.cut(u'我认为大家还是创业者、创业家，其实整个社会的支持和关注下成长起来的'))]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), fontproperties=myfont)
    print(word)

pyplot.xlabel("横轴", fontproperties=myfont)
pyplot.ylabel("纵轴", fontproperties=myfont)
pyplot.title("pythoner.com", fontproperties=myfont)
legend(['图例'], prop=myfont)

pyplot.show()
