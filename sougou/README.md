from
http://www.jianshu.com/p/6d542ff65b1e

#
仅支持python3
#
1、将 xml 格式语料文件转换为 txt 格式
~~~shell
cat news_tensite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>"  > corpus.txt 
~~~

2、用jieba 分词将文本文件分词
~~~python
python word_segment.py corpus.txt corpus_seg.txt
~~~

3、用Word2Vec 训练
~~~python
python train_word2vec_model.py corpus_seg.txt corpus.model corpus.vector
~~~

4、测试训练结果
~~~shell
from gensim.models import Word2Vec
model = Word2Vec.load('corpus.model')

result = model.most_similar(u'妹纸')
print(result)
~~~