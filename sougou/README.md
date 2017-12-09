from
http://www.jianshu.com/p/6d542ff65b1e

#
1、
~~~shell
cat news_tensite_xml.dat | iconv -f gbk -t utf-8 -c | grep "<content>"  > corpus.txt 
~~~

2、
~~~python
python word_segment.py corpus.txt corpus_seg.txt
~~~

3、
~~~python
python train_word2vec_model.py corpus_seg.txt corpus.model corpus.vector
~~~